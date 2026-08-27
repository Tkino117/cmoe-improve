"""方式4: 回収率を共同最適化した代表集合。

方式1〜3 は代表を expert ごとに独立に選ぶ。しかしルーティングはトークンごとの
**expert 間の比較**であり、Top-K を決めるのは score の相対順位だけである。
1 expert ずつ評価する基準は、制御しようとしている量そのものを見られない。

この方式は回収率そのものを最大化する:

    R = Σ_t (M_shared(t) + Σ_{j∈TopK(q_t)} M_j(t)) / Σ_t (M_shared(t) + Σ_j M_j(t))

配備できる族（1 expert・1実在ニューロン、分割そのまま、L2 正規化そのまま）の
中で探す。

探索は座標上昇で、1座標は厳密かつ安価である。他の expert の score を固定し、
その K 番目に大きい値を b_k(t) とすると、候補 score v が Top-K に入る条件は
v > b_k(t) だけで、そのとき増える質量は M_e(t) − M(b_k(t))。**どの候補が v を
出したかに依存しない**ので、目的関数は

    const + Σ_t 1[x_j(t) > b_k(t)] · delta(t)

となり、expert 内の全候補を1回の縮約で同時に評価できる。

受理した手は毎回 Top-K で測り直し、R が本当に増えたときだけ残す。上の縮約は
v == b_k(t) を「選ばれない」と解くが torch.topk の同点の解き方は別なので、
測り直すことで単調性を仮定ではなく構成で保証する。

CMoE-ref の ``routerlab/methods/oracle_recovery.py`` の移送。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cmoe.router.methods.frequency_centroid import install_representatives
from cmoe.router.methods.oracle_correlation import (activation, flatten_z,
                                                    validate_weights)


def clean_group(group, n_neurons, label):
    group = tuple(int(index) for index in group)
    if not group:
        raise ValueError(f'{label} が空')
    if len(set(group)) != len(group):
        raise ValueError(f'{label} にニューロンの重複がある')
    if min(group) < 0 or max(group) >= n_neurons:
        raise ValueError(f'{label} に 0..{n_neurons - 1} の外の番号がある')
    return group


@dataclass(frozen=True)
class RecoverySelection:
    """選ばれた集合、その回収率、そこへ至った経路。"""

    representatives: tuple
    recovery: float
    initial_recoveries: tuple
    starts: tuple
    chosen_start: int
    candidate_counts: tuple
    n_tokens: int

    def metadata(self):
        return {
            'representatives': list(self.representatives),
            'fit_recovery': self.recovery,
            'initial_recoveries': list(self.initial_recoveries),
            'chosen_start': self.chosen_start,
            'candidate_counts': list(self.candidate_counts),
            'n_tokens': self.n_tokens,
            'starts': [
                {
                    'start_index': row['start_index'],
                    'initial_representatives': list(row['initial_representatives']),
                    'initial_recovery': row['initial_recovery'],
                    'final_representatives': list(row['final_representatives']),
                    'final_recovery': row['final_recovery'],
                    'sweeps_used': row['sweeps_used'],
                    'termination': row['termination'],
                    'accepted_moves': [dict(move) for move in row['accepted_moves']],
                }
                for row in self.starts
            ],
        }


@torch.no_grad()
def expert_masses(z, groups, gate_weight, up_weight, chunk_size):
    """トークンごとの真の |h| 質量: shared グループと各 routed グループ。"""
    device = gate_weight.device
    n_tokens = z.shape[0]
    tensors = [torch.tensor(list(group), dtype=torch.long, device=device)
               for group in groups]
    routed = torch.empty((n_tokens, len(groups) - 1), dtype=torch.float32, device=device)
    shared = torch.zeros(n_tokens, dtype=torch.float32, device=device)
    for start in range(0, n_tokens, chunk_size):
        stop = min(start + chunk_size, n_tokens)
        chunk = z[start:stop].to(device)
        for index, indices in enumerate(tensors):
            if not indices.numel():
                continue
            mass = activation(
                chunk,
                gate_weight.index_select(0, indices),
                up_weight.index_select(0, indices),
            ).abs_().sum(dim=1, dtype=torch.float32)
            if index == 0:
                shared[start:stop] = mass
            else:
                routed[start:stop, index - 1] = mass
    return routed, shared


@torch.no_grad()
def _candidate_scores(z, group, norm_gate, norm_up, chunk_size, dtype):
    """全 fit トークンに対する、各候補ニューロンの配備 router score。"""
    device = norm_gate.device
    indices = torch.tensor(list(group), dtype=torch.long, device=device)
    scores = torch.empty((z.shape[0], indices.numel()), dtype=dtype, device=device)
    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        chunk = z[start:stop].to(device)
        scores[start:stop] = activation(
            chunk,
            norm_gate.index_select(0, indices),
            norm_up.index_select(0, indices),
        ).abs_().to(dtype)
    return scores


def _selected_mass(scores, columns, routed, topk):
    picked = torch.stack(
        [scores[expert][:, column].float()
         for expert, column in enumerate(columns)], dim=1)
    indices = picked.topk(topk, dim=1).indices
    return float(routed.gather(1, indices).sum(dtype=torch.float64))


def _kth_of_others(scores, columns, routed, expert, topk):
    """他の expert の K 番目に大きい score と、それと入れ替わる質量。"""
    others = [index for index in range(len(columns)) if index != expert]
    picked = torch.stack(
        [scores[index][:, columns[index]].float() for index in others], dim=1)
    order = picked.argsort(dim=1, descending=True)
    position = order[:, topk - 1:topk]
    threshold = picked.gather(1, position).squeeze(1)
    displaced = routed[:, others].gather(1, position).squeeze(1)
    return threshold, routed[:, expert] - displaced


@torch.no_grad()
def _best_candidate(scores, expert, threshold, delta, pool, neurons, token_chunk):
    """1 expert の厳密な最適応答。同点なら最小のニューロン番号。"""
    total = torch.zeros(scores[expert].shape[1], dtype=torch.float64,
                        device=scores[expert].device)
    for start in range(0, scores[expert].shape[0], token_chunk):
        stop = min(start + token_chunk, scores[expert].shape[0])
        selected = (scores[expert][start:stop].float()
                    > threshold[start:stop, None]).float()
        selected.mul_(delta[start:stop, None])
        total.add_(selected.sum(dim=0, dtype=torch.float64))
    if pool is not None:
        allowed = torch.full_like(total, -float('inf'))
        allowed[pool] = total[pool]
        total = allowed
    best = float(total.max())
    tied = (total == best).nonzero(as_tuple=True)[0].tolist()
    return min(tied, key=lambda column: neurons[column])


@dataclass(frozen=True)
class _Search:
    scores: list
    routed: torch.Tensor
    shared_mass: float
    total: float
    routed_groups: list
    positions: list
    pool_columns: object
    starts: list
    n_tokens: int


@torch.no_grad()
def _prepare_search(expert_groups, z, gate_weight, up_weight, initial_sets,
                    router_normalized, chunk_size, candidate_pools, score_dtype):
    """入力を検査し、掃引が読み直すものを実体化する。"""
    validate_weights(gate_weight, up_weight)
    if chunk_size < 1:
        raise ValueError('chunk_size は正のはず')
    z = flatten_z(z, gate_weight.shape[1])
    n_neurons = gate_weight.shape[0]
    n_tokens = z.shape[0]
    device = gate_weight.device

    groups = [()]
    if expert_groups and expert_groups[0]:
        groups[0] = clean_group(expert_groups[0], n_neurons, 'shared グループ')
    routed_groups = [clean_group(group, n_neurons, f'routed expert {index}')
                     for index, group in enumerate(expert_groups[1:])]
    groups.extend(routed_groups)
    n_routed = len(routed_groups)
    if n_routed < 1:
        raise ValueError('回収率探索には routed expert が1個以上要る')

    if not initial_sets:
        raise ValueError('回収率探索には初期代表集合が1つ以上要る')
    starts_input = []
    for row in initial_sets:
        row = tuple(int(index) for index in row)
        if len(row) != n_routed:
            raise ValueError(f'初期集合 {row} が routed expert {n_routed} 個を覆わない')
        for expert, neuron in enumerate(row):
            if neuron not in routed_groups[expert]:
                raise ValueError(
                    f'初期代表 {neuron} が routed expert {expert} の中にいない')
        if row not in starts_input:
            starts_input.append(row)

    pools = None
    if candidate_pools:
        if len(candidate_pools) != n_routed:
            raise ValueError(
                f'候補プールが {len(candidate_pools)} 個、routed expert は {n_routed} 個')
        pools = []
        for expert, pool in enumerate(candidate_pools):
            pool = clean_group(pool, n_neurons, f'候補プール {expert}')
            members = set(routed_groups[expert])
            outside = [neuron for neuron in pool if neuron not in members]
            if outside:
                raise ValueError(
                    f'候補プール {expert} に expert 外のニューロンがある: {outside[:4]}')
            for row in starts_input:
                if row[expert] not in pool:
                    raise ValueError(
                        f'候補プール {expert} が初期代表 {row[expert]} を除いている')
            pools.append(pool)

    score_gate, score_up = gate_weight, up_weight
    if router_normalized:
        score_gate = F.normalize(score_gate, p=2, dim=1)
        score_up = F.normalize(score_up, p=2, dim=1)

    routed, shared = expert_masses(z, groups, gate_weight, up_weight, chunk_size)
    total = float((shared + routed.sum(dim=1)).sum(dtype=torch.float64))
    if not total > 0:
        raise ValueError(f'fit の活性量が {total}')
    scores = [_candidate_scores(z, group, score_gate, score_up, chunk_size, score_dtype)
              for group in routed_groups]
    positions = [{neuron: column for column, neuron in enumerate(group)}
                 for group in routed_groups]
    pool_columns = None
    if pools is not None:
        pool_columns = [torch.tensor([positions[expert][neuron] for neuron in pool],
                                     dtype=torch.long, device=device)
                        for expert, pool in enumerate(pools)]

    return _Search(
        scores=scores,
        routed=routed,
        shared_mass=float(shared.sum(dtype=torch.float64)),
        total=total,
        routed_groups=routed_groups,
        positions=positions,
        pool_columns=pool_columns,
        starts=starts_input,
        n_tokens=n_tokens,
    )


@torch.no_grad()
def select_recovery_representatives(
        expert_groups, topk, z, gate_weight, up_weight, initial_sets,
        router_normalized=True, chunk_size=4096, token_chunk=8192,
        max_sweeps=10, candidate_pools=(), score_dtype=torch.float32):
    """既知で最良の単一代表集合を座標上昇で探す。

    ``expert_groups`` は先頭に shared グループを含む。``initial_sets`` は先行方式
    の代表集合で、相異なるものそれぞれから探索して最良を採るので、**出発点に
    した方式より悪い集合を返すことはない**。
    """
    if token_chunk < 1 or max_sweeps < 1:
        raise ValueError('token_chunk と max_sweeps は正のはず')
    search = _prepare_search(
        expert_groups, z, gate_weight, up_weight, initial_sets,
        router_normalized, chunk_size, candidate_pools, score_dtype)
    scores, routed = search.scores, search.routed
    routed_groups, positions = search.routed_groups, search.positions
    pool_columns, total = search.pool_columns, search.total
    shared_mass = search.shared_mass
    n_routed = len(routed_groups)
    if not isinstance(topk, int) or isinstance(topk, bool) or not 0 <= topk <= n_routed:
        raise ValueError(f'Top-K {topk!r} は routed expert {n_routed} 個に対して不正')
    # Top-K が 0 か全部なら、どの代表を選んでも選択は変わらない
    frozen = topk == 0 or topk == n_routed

    records = []
    for start_index, row in enumerate(search.starts):
        columns = [positions[expert][row[expert]] for expert in range(n_routed)]
        recovery = (shared_mass + _selected_mass(scores, columns, routed, topk)) / total
        initial_recovery = recovery
        moves = []
        sweeps_used = 0
        termination = 'objective-independent-of-selection' if frozen else 'converged'
        if not frozen:
            for sweep in range(max_sweeps):
                sweeps_used = sweep + 1
                changed = False
                for expert in range(n_routed):
                    threshold, delta = _kth_of_others(
                        scores, columns, routed, expert, topk)
                    column = _best_candidate(
                        scores, expert, threshold, delta,
                        None if pool_columns is None else pool_columns[expert],
                        routed_groups[expert], token_chunk)
                    if column == columns[expert]:
                        continue
                    trial = list(columns)
                    trial[expert] = column
                    value = (shared_mass
                             + _selected_mass(scores, trial, routed, topk)) / total
                    # 縮約は同点の解き方までしか厳密でないので、直接測って残らな
                    # かった「改善」は雑音として捨てる
                    if value <= recovery:
                        continue
                    moves.append({
                        'sweep': sweep,
                        'expert': expert,
                        'from_neuron': routed_groups[expert][columns[expert]],
                        'to_neuron': routed_groups[expert][column],
                        'recovery': value,
                    })
                    columns, recovery, changed = trial, value, True
                if not changed:
                    break
            else:
                termination = 'max-sweeps'
        records.append({
            'start_index': start_index,
            'initial_representatives': row,
            'initial_recovery': initial_recovery,
            'final_representatives': tuple(
                routed_groups[expert][columns[expert]] for expert in range(n_routed)),
            'final_recovery': recovery,
            'sweeps_used': sweeps_used,
            'termination': termination,
            'accepted_moves': moves,
        })

    best = max(records, key=lambda row: (row['final_recovery'],
                                         tuple(-n for n in row['final_representatives'])))
    return RecoverySelection(
        representatives=best['final_representatives'],
        recovery=best['final_recovery'],
        initial_recoveries=tuple(row['initial_recovery'] for row in records),
        starts=tuple(records),
        chosen_start=best['start_index'],
        candidate_counts=tuple(
            len(group) if pool_columns is None else int(pool_columns[expert].numel())
            for expert, group in enumerate(routed_groups)),
        n_tokens=search.n_tokens,
    )


class OracleRecoveryMethod:
    name = 'oracle_recovery'
    training_free = True
    requires_source_weights = True
    requires_fit_z = True
    requires_initial_sets = True
    diagnostic_only = False

    def __init__(self, chunk_size=4096, token_chunk=8192, max_sweeps=10):
        self.chunk_size = chunk_size
        self.token_chunk = token_chunk
        self.max_sweeps = max_sweeps

    def build(self, context, baseline):
        selection = select_recovery_representatives(
            context.expert_groups,
            context.topk,
            context.fit_z,
            context.gate_weight,
            context.up_weight,
            context.initial_representative_sets,
            router_normalized=context.router_normalized,
            chunk_size=self.chunk_size,
            token_chunk=self.token_chunk,
            max_sweeps=self.max_sweeps,
            candidate_pools=context.candidate_pools,
        )
        return install_representatives(
            baseline, context, selection.representatives,
            {'rule': 'coordinate-ascent-on-true-abs-mass-recovery',
             **selection.metadata()})
