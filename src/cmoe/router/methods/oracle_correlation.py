"""方式3: expert 全体の真の |h| 活性量と最も相関する1ニューロンを代表にする。

方式1・2 が「クラスタの中心らしさ」で代表を選ぶのに対し、この方式は router の
目的に直接合わせる — 代表 score が expert 全体の活性量をどれだけ言い当てるか
(Pearson 相関) が最大のニューロンを選ぶ。

候補 score は配備される router の式そのもの（L2 正規化込み）で計算する。十分
統計量しか持たないので、64×2048 トークンでも H 行列を実体化しない。

CMoE-ref の ``routerlab/methods/oracle_correlation.py`` の移送。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cmoe.router.methods.frequency_centroid import install_representatives


def flatten_z(z, hidden_size):
    if z is None or z.shape[-1] != hidden_size:
        shape = None if z is None else tuple(z.shape)
        raise ValueError(f'fit z の形が {shape}、最終次元 {hidden_size} のはず')
    z = z.reshape(-1, hidden_size)
    if z.shape[0] < 2:
        raise ValueError('Pearson 相関には2トークン以上要る')
    if not bool(torch.isfinite(z).all()):
        raise ValueError('fit z は有限のはず')
    return z


def validate_weights(gate_weight, up_weight):
    if gate_weight is None or up_weight is None:
        raise ValueError('この方式は dense の gate/up 重みを要求する')
    if gate_weight.dim() != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError(
            f'gate/up 重みは共通の2次元の形のはず。{tuple(gate_weight.shape)} と '
            f'{tuple(up_weight.shape)} を受け取った')


def group_indices(group, n_neurons, device):
    group = tuple(int(index) for index in group)
    if not group:
        raise ValueError('expert グループが空')
    if len(set(group)) != len(group):
        raise ValueError('expert グループにニューロンの重複がある')
    if min(group) < 0 or max(group) >= n_neurons:
        raise ValueError(f'expert グループに 0..{n_neurons - 1} の外の番号がある')
    return group, torch.tensor(group, dtype=torch.long, device=device)


def activation(z, gate, up):
    return F.silu(F.linear(z, gate)) * F.linear(z, up)


def pearson_from_sums(n, sum_x, sum_x2, sum_xy, sum_y, sum_y2,
                      min_x, max_x, min_y, max_y):
    constant = min_x == max_x
    if min_y == max_y:
        raise ValueError('expert の |h| 活性量が fit データ上で定数')
    numerator = n * sum_xy - sum_x * sum_y
    var_x = (n * sum_x2 - sum_x.square()).clamp_min_(0)
    var_y = max(float(n * sum_y2 - sum_y * sum_y), 0.0)
    if not var_y > 0:
        raise ValueError('expert の |h| 活性量の数値的分散が 0')
    correlations = numerator / torch.sqrt(var_x * var_y)
    correlations[constant] = torch.nan
    return correlations, constant


@dataclass(frozen=True)
class CorrelationSelection:
    representatives: tuple
    correlations: tuple
    top_candidates: tuple
    excluded_constant_scores: tuple
    n_tokens: int

    def metadata(self):
        return {
            'representatives': list(self.representatives),
            'correlations': list(self.correlations),
            'top_candidates': [
                [{'neuron': neuron, 'correlation': correlation}
                 for neuron, correlation in ranking]
                for ranking in self.top_candidates
            ],
            'excluded_constant_scores': list(self.excluded_constant_scores),
            'n_tokens': self.n_tokens,
        }


@torch.no_grad()
def select_correlated_representatives(routed_groups, z, gate_weight, up_weight,
                                      router_normalized=True, chunk_size=4096,
                                      keep_top=16):
    """expert ごとに、真の |h| 活性量との Pearson 相関が最大の実在ニューロン。"""
    validate_weights(gate_weight, up_weight)
    if chunk_size < 1 or keep_top < 1:
        raise ValueError('chunk_size と keep_top は正のはず')
    z = flatten_z(z, gate_weight.shape[1])
    n_tokens = z.shape[0]
    n_neurons = gate_weight.shape[0]
    device = gate_weight.device
    score_gate = gate_weight
    score_up = up_weight
    if router_normalized:
        score_gate = F.normalize(score_gate, p=2, dim=1)
        score_up = F.normalize(score_up, p=2, dim=1)

    representatives = []
    selected_correlations = []
    top_candidates = []
    excluded_counts = []
    for expert_index, raw_group in enumerate(routed_groups):
        group, indices = group_indices(raw_group, n_neurons, device)
        raw_gate = gate_weight.index_select(0, indices)
        raw_up = up_weight.index_select(0, indices)
        candidate_gate = score_gate.index_select(0, indices)
        candidate_up = score_up.index_select(0, indices)
        size = len(group)
        sum_x = torch.zeros(size, dtype=torch.float64, device=device)
        sum_x2 = torch.zeros_like(sum_x)
        sum_xy = torch.zeros_like(sum_x)
        sum_y = torch.zeros((), dtype=torch.float64, device=device)
        sum_y2 = torch.zeros_like(sum_y)
        min_x = torch.full((size,), math.inf, dtype=torch.float32, device=device)
        max_x = torch.full((size,), -math.inf, dtype=torch.float32, device=device)
        min_y = math.inf
        max_y = -math.inf

        for start in range(0, n_tokens, chunk_size):
            chunk = z[start:start + chunk_size].to(device)
            target = activation(chunk, raw_gate, raw_up).abs_().sum(
                dim=1, dtype=torch.float32)
            scores = activation(chunk, candidate_gate, candidate_up).abs_().float()
            sum_x.add_(scores.sum(dim=0, dtype=torch.float64))
            sum_x2.add_(scores.square().sum(dim=0, dtype=torch.float64))
            sum_xy.add_((scores * target[:, None]).sum(dim=0, dtype=torch.float64))
            sum_y.add_(target.sum(dtype=torch.float64))
            sum_y2.add_(target.square().sum(dtype=torch.float64))
            min_x.copy_(torch.minimum(min_x, scores.amin(dim=0)))
            max_x.copy_(torch.maximum(max_x, scores.amax(dim=0)))
            min_y = min(min_y, float(target.min()))
            max_y = max(max_y, float(target.max()))

        correlations, constant = pearson_from_sums(
            n_tokens, sum_x, sum_x2, sum_xy, sum_y, sum_y2,
            min_x, max_x, min_y, max_y)
        eligible = []
        for neuron, correlation, excluded in zip(
                group, correlations.tolist(), constant.tolist()):
            if excluded:
                continue
            if not math.isfinite(correlation):
                raise ValueError(
                    f'expert {expert_index} のニューロン {neuron} の相関が非有限')
            eligible.append((neuron, correlation))
        if not eligible:
            raise ValueError(f'expert {expert_index} に定数でない候補 score が無い')
        eligible.sort(key=lambda item: (-item[1], item[0]))
        winner = eligible[0]
        representatives.append(winner[0])
        selected_correlations.append(winner[1])
        top_candidates.append(tuple(eligible[:keep_top]))
        excluded_counts.append(int(constant.sum()))

    return CorrelationSelection(
        representatives=tuple(representatives),
        correlations=tuple(selected_correlations),
        top_candidates=tuple(top_candidates),
        excluded_constant_scores=tuple(excluded_counts),
        n_tokens=n_tokens,
    )


class OracleCorrelationMethod:
    name = 'oracle_correlation'
    training_free = True
    requires_source_weights = True
    requires_fit_z = True
    diagnostic_only = False

    def __init__(self, chunk_size=4096, keep_top=16):
        self.chunk_size = chunk_size
        self.keep_top = keep_top

    def build(self, context, baseline):
        selection = select_correlated_representatives(
            context.routed_groups,
            context.fit_z,
            context.gate_weight,
            context.up_weight,
            router_normalized=context.router_normalized,
            chunk_size=self.chunk_size,
            keep_top=self.keep_top,
        )
        return install_representatives(
            baseline, context, selection.representatives,
            {'rule': 'maximum-pearson-correlation-with-true-absolute-expert-mass',
             **selection.metadata()})
