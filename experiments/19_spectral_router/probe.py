"""落とした材料の上で、ルーターの score 関数を並べて採点する。

モデルを走らせないので1案あたり数秒で回る。ここで見るのは4つ。

* ``exact``  … オラクル ``oracle_abs`` と Top-K 集合が完全一致した割合
* ``recall`` … 同じく、選んだ expert のうちオラクルと重なった割合
* ``mass``   … 選んだ expert が回収した |h| 質量の割合（診断の ``router_r``）
* ``err``    … その層の出力誤差 ‖y_dense − y_sel‖ / ‖y_dense‖

``mass`` は既存の診断と同じ量、``err`` は配分オラクル ``local_error`` と同じ量で
ある。**採否は最終的にベンチが決める**（report/10 で回収率が上がって acc が
下がった前例があり、report/19 では `err` が最良の案が PPL で負けた）ので、ここは
案を絞るための道具であって結論ではない。

    uv run python experiments/19_spectral_router/probe.py --scorers cmoe,topm8,wlowrank32
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

DEFAULT_DIR = os.path.join(
    os.environ.get('CMOE_PROBE_DIR', '/tmp/claude-1000/probe'), 'dump')


class Material:
    """1層分の材料。重みは GPU に、z はホストに置く。"""

    def __init__(self, payload, device):
        self.device = device
        self.layer = int(payload['layer'])
        self.topk = int(payload['topk'])
        self.n_shared = int(payload['n_shared'])
        self.groups = [torch.tensor(group, dtype=torch.long, device=device)
                       for group in payload['expert_groups']]
        self.representatives = list(payload['representatives'])
        self.rates = payload['rates'].to(device)
        self.gate = payload['gate_weight'].to(device, torch.float32)
        self.up = payload['up_weight'].to(device, torch.float32)
        self.down = payload['down_weight'].to(device, torch.float32)
        self.fit_z = payload['fit_z'].to(torch.float32)
        self.validation_z = payload['validation_z'].to(torch.float32)
        self.hidden_size = self.gate.shape[1]

    @property
    def routed(self):
        return self.groups[1:]

    @property
    def n_routed(self):
        return len(self.groups) - 1

    def expert_rows(self, index):
        """expert index（0 が shared）の gate / up / down 行。"""
        rows = self.groups[index]
        return (self.gate.index_select(0, rows),
                self.up.index_select(0, rows),
                self.down.index_select(1, rows))

    def h(self, x, index):
        """expert index の中間活性 [tokens, neurons]。"""
        gate, up, _ = self.expert_rows(index)
        return F.silu(F.linear(x, gate)).mul_(F.linear(x, up))


def batches(z, size, device):
    for start in range(0, z.shape[0], size):
        yield z[start:start + size].to(device)


# --- score 関数 -----------------------------------------------------------
# 各方式は Material と fit の z を受け取り、``callable(x) -> [tokens, n_routed]``
# と、1トークンあたりのルーター MAC 数を返す。


def scorer_cmoe(material, fit, spec):
    """現行 CMoE。代表ニューロン1本、行は L2 正規化。"""
    rows = torch.tensor(material.representatives, dtype=torch.long,
                        device=material.device)
    gate = F.normalize(material.gate.index_select(0, rows), p=2, dim=1)
    up = F.normalize(material.up.index_select(0, rows), p=2, dim=1)

    def score(x):
        return (F.linear(x, up) * F.silu(F.linear(x, gate))).abs()

    return score, 2 * material.hidden_size * material.n_routed


def scorer_cmoe_raw(material, fit, spec):
    """代表1本、正規化なし（素の行）。"""
    rows = torch.tensor(material.representatives, dtype=torch.long,
                        device=material.device)
    gate = material.gate.index_select(0, rows)
    up = material.up.index_select(0, rows)

    def score(x):
        return (F.linear(x, up) * F.silu(F.linear(x, gate))).abs()

    return score, 2 * material.hidden_size * material.n_routed


def mean_abs_h(material, fit, index, chunk=4096):
    """expert index の各ニューロンの平均 |h|（fit 上）。"""
    total = None
    count = 0
    for x in batches(fit, chunk, material.device):
        value = material.h(x, index).abs_().sum(dim=0, dtype=torch.float32)
        total = value if total is None else total + value
        count += x.shape[0]
    return total / count


def pick_rows(material, fit, mode, m, seed=0):
    """expert ごとに m 本のニューロン行を選ぶ。"""
    picks = []
    for index in range(1, len(material.groups)):
        group = material.groups[index]
        size = group.numel()
        take = min(m, size)
        if mode == 'top':
            order = mean_abs_h(material, fit, index).argsort(descending=True)
            picks.append(group.index_select(0, order[:take]))
        elif mode == 'rand':
            generator = torch.Generator(device='cpu').manual_seed(
                seed * 1000 + index + material.layer)
            order = torch.randperm(size, generator=generator).to(group.device)
            picks.append(group.index_select(0, order[:take]))
        elif mode == 'spread':
            # 平均 |h| の順に並べてから等間隔に拾う（頭と裾を混ぜる）
            order = mean_abs_h(material, fit, index).argsort(descending=True)
            step = max(1, size // take)
            picks.append(group.index_select(0, order[::step][:take]))
        else:
            raise ValueError(f'未知の選び方 {mode!r}')
    return picks


def scorer_rows(material, fit, spec):
    """m 本のニューロンの |h| を足して expert の質量を見積もる。

    ``topm8`` = 平均 |h| の上位8本、``randm8`` = 無作為8本、
    ``spreadm8`` = 順位の等間隔8本。
    """
    mode = {'topm': 'top', 'randm': 'rand', 'spreadm': 'spread'}[spec['kind']]
    m = spec['m']
    picks = pick_rows(material, fit, mode, m)
    gate = torch.stack([material.gate.index_select(0, rows) for rows in picks])
    up = torch.stack([material.up.index_select(0, rows) for rows in picks])
    # 無作為抽出は不偏推定にするため expert の大きさで割り戻す（順位は
    # expert ごとの倍率で変わるので、これは効く）
    if mode == 'rand':
        scale = torch.tensor(
            [group.numel() / rows.numel()
             for group, rows in zip(material.routed, picks)],
            dtype=torch.float32, device=material.device)
    else:
        scale = torch.ones(len(picks), dtype=torch.float32,
                           device=material.device)

    def score(x):
        values = []
        for index in range(len(picks)):
            h = F.silu(F.linear(x, gate[index])).mul_(F.linear(x, up[index]))
            values.append(h.abs_().sum(dim=1))
        return torch.stack(values, dim=1) * scale

    return score, 2 * m * material.hidden_size * material.n_routed


def data_weighted_basis(material, fit, weight, rank, chunk=4096):
    """‖W x‖² を最もよく説明する rank 本の方向。

    fit の上での E[(W x)(W x)ᵀ] ではなく、E[‖W x‖²] を分解したいので、
    x の側の2次モーメントで重みを付けた WᵀW の主部分を取る。
    ``M = Σ_x (W x)(W x)ᵀ`` の代わりに ``C = Wᵀ W`` を x の共分散で
    白色化してから固有分解する。
    """
    covariance = torch.zeros(material.hidden_size, material.hidden_size,
                             dtype=torch.float64, device=material.device)
    count = 0
    for x in batches(fit, chunk, material.device):
        covariance += (x.T @ x).double()
        count += x.shape[0]
    covariance /= count
    # 白色化 x = L u（L は共分散のコレスキー）。‖W L u‖² の主方向を取れば、
    # データの上での寄与が大きい順になる
    jitter = 1e-4 * float(torch.diagonal(covariance).mean())
    chol = torch.linalg.cholesky(
        covariance + jitter * torch.eye(material.hidden_size, dtype=torch.float64,
                                        device=material.device))
    projected = (weight.double() @ chol).float()
    _, singular, right = torch.linalg.svd(projected, full_matrices=False)
    take = min(rank, right.shape[0])
    # u 空間の方向を x 空間へ戻す
    directions = torch.linalg.solve_triangular(
        chol.T.float(), right[:take].T, upper=True).T
    return directions * singular[:take, None], float(
        (singular[:take].square().sum() / singular.square().sum()).item())


def silu_row_scale(material, fit, index, chunk=4096):
    """gate 行ごとの ``E[σ(a)²]``。silu の非対称性を行の重みに畳む。

    ``Σ_i silu(a_i)² = Σ_i σ(a_i)² a_i²`` なので、σ(a_i)² を行ごとの定数で
    置き換えれば ``‖diag(√p) G_e x‖²`` という2次形式に戻る。ふだん負に居る
    ニューロンはここで小さくなり、低ランク近似の予算がそちらへ流れなくなる。
    """
    gate, _, _ = material.expert_rows(index)
    total = None
    count = 0
    for x in batches(fit, chunk, material.device):
        a = F.linear(x, gate)
        value = torch.sigmoid(a).square_().sum(dim=0, dtype=torch.float32)
        total = value if total is None else total + value
        count += x.shape[0]
    return (total / count).sqrt()


def scorer_lowrank(material, fit, spec):
    """‖G_e x‖·‖U_e x‖ を rank r で近似する（Cauchy–Schwarz の上界）。

    ``wlowrank`` は gate 行に ``√E[σ(a)²]`` を掛けてから分解する版で、
    ``cssilu``（silu を通してからノルムを取る形）へ寄せた近似になる。
    """
    rank = spec['m']
    kind = spec['kind']
    weighted = 'w' in kind.replace('lowrank', '')
    # ``o`` が付く版は down_proj の列ノルムも行の重みに掛ける。近似する量が
    # 「質量 Σ|h_i|」から「出力への寄与 Σ‖d_i‖|h_i|」に変わる
    output_weighted = 'o' in kind.replace('lowrank', '')
    # ``c`` が付く版は expert ごとの倍率を校正で合わせる
    calibrated = 'c' in kind.replace('lowrank', '')
    # ``p`` が付く版は白色化しない素の SVD。校正データを基底に使わない
    plain = kind.startswith('p')

    def basis(weight):
        if plain:
            _, singular, right = torch.linalg.svd(weight, full_matrices=False)
            take = min(rank, right.shape[0])
            # ‖Wx‖² ≈ Σ σ² (v·x)² なので、方向だけでなく特異値も要る
            return right[:take] * singular[:take, None]
        return data_weighted_basis(material, fit, weight, rank)[0]

    gate_basis, up_basis = [], []
    for index in range(1, len(material.groups)):
        gate, up, _ = material.expert_rows(index)
        if weighted:
            gate = gate * silu_row_scale(material, fit, index)[:, None]
        if output_weighted:
            _, _, down = material.expert_rows(index)
            gate = gate * down.norm(dim=0)[:, None]
        gate_basis.append(basis(gate))
        up_basis.append(basis(up))
    gate_basis = torch.stack(gate_basis)
    up_basis = torch.stack(up_basis)

    def raw(x):
        values = []
        for index in range(gate_basis.shape[0]):
            a = F.linear(x, gate_basis[index]).square().sum(dim=1)
            b = F.linear(x, up_basis[index]).square().sum(dim=1)
            values.append((a * b).sqrt())
        return torch.stack(values, dim=1)

    if not calibrated:
        return raw, 2 * rank * material.hidden_size * material.n_routed

    # Cauchy–Schwarz の緩みは expert ごとに違う（行の中で |silu(a)| と |b| が
    # どれだけ揃っているかで決まる）。その系統的なずれだけを、真の質量との
    # 平均比という1 expert あたり1個のスカラーで消す
    numerator = torch.zeros(material.n_routed, dtype=torch.float64,
                            device=material.device)
    denominator = torch.zeros_like(numerator)
    for x in batches(fit, 4096, material.device):
        numerator += true_mass(material, x).sum(dim=0, dtype=torch.float64)
        denominator += raw(x).sum(dim=0, dtype=torch.float64)
    alpha = (numerator / denominator.clamp_min(1e-12)).float()

    def score(x):
        return raw(x) * alpha

    return score, 2 * rank * material.hidden_size * material.n_routed


def scorer_norms(material, fit, spec):
    """行列ノルムだけで質量を見積もる（full rank。関数形の天井を見るため）。

    ``csfull`` = ‖G_e x‖·‖U_e x‖、``gnormfull`` = ‖G_e x‖、
    ``unormfull`` = ‖U_e x‖。どれも学習も校正も要らず、重みだけから決まる。
    """
    kind = spec['kind']
    scales = ({index: silu_row_scale(material, fit, index)
               for index in range(1, len(material.groups))}
              if kind == 'wcsfull' else {})

    def score(x):
        values = []
        for index in range(1, len(material.groups)):
            gate, up, _ = material.expert_rows(index)
            if index in scales:
                gate = gate * scales[index][:, None]
            if kind == 'gnormfull':
                values.append(F.linear(x, gate).norm(dim=1))
            elif kind == 'unormfull':
                values.append(F.linear(x, up).norm(dim=1))
            else:
                values.append(F.linear(x, gate).norm(dim=1)
                              * F.linear(x, up).norm(dim=1))
        return torch.stack(values, dim=1)

    total = sum(group.numel() for group in material.routed)
    return score, 2 * total * material.hidden_size


def scorer_nonlinear_norms(material, fit, spec):
    """silu を通してからノルムを取る形（関数形の天井を見るため）。

    ``csrelu`` = ‖relu(G_e x)‖·‖U_e x‖、``cssilu`` = ‖silu(G_e x)‖·‖U_e x‖。
    どちらも低ランク化できない（非線形が射影と交換しない）ので、これは
    「もっと良い形があるか」を測るためだけの行である。
    """
    kind = spec['kind']

    def score(x):
        values = []
        for index in range(1, len(material.groups)):
            gate, up, _ = material.expert_rows(index)
            a = F.linear(x, gate)
            a = F.relu(a) if kind == 'csrelu' else F.silu(a)
            values.append(a.norm(dim=1) * F.linear(x, up).norm(dim=1))
        return torch.stack(values, dim=1)

    total = sum(group.numel() for group in material.routed)
    return score, 2 * total * material.hidden_size


def true_mass(material, x):
    """[tokens, n_routed] の真の |h| 質量。"""
    values = []
    for index in range(1, len(material.groups)):
        values.append(material.h(x, index).abs_().sum(dim=1))
    return torch.stack(values, dim=1)


def scorer_oracle(material, fit, spec):
    def score(x):
        return true_mass(material, x)

    total = sum(group.numel() for group in material.routed)
    return score, 2 * total * material.hidden_size


def scorer_oracle_out(material, fit, spec):
    """真の出力寄与 ‖down_e(h_e)‖ で選ぶ（別のオラクル）。"""

    def score(x):
        values = []
        for index in range(1, len(material.groups)):
            _, _, down = material.expert_rows(index)
            h = material.h(x, index)
            values.append((h @ down.T).norm(dim=1))
        return torch.stack(values, dim=1)

    total = sum(group.numel() for group in material.routed)
    return score, 3 * total * material.hidden_size


# 名前の接頭辞が低ランクの作り方を決める。``w`` = silu を行の重みに畳む、
# ``p`` = 白色化しない素の SVD、``o`` = down_proj の列ノルムも掛ける、
# ``c`` = expert ごとの倍率を真の質量に合わせる。組み合わせて書ける
# （``pwlowrank32`` = 白色化なし・silu 重みあり・rank 32）。
#
# ここに残っているのは report/19 に載った案だけである。落とした案とその数字は
# README にある。
SCORERS = {
    'cmoe': scorer_cmoe,
    'cmoe_raw': scorer_cmoe_raw,
    'topm': scorer_rows,
    'randm': scorer_rows,
    'spreadm': scorer_rows,
    'lowrank': scorer_lowrank,
    'wlowrank': scorer_lowrank,
    'plowrank': scorer_lowrank,
    'pwlowrank': scorer_lowrank,
    'owlowrank': scorer_lowrank,
    'olowrank': scorer_lowrank,
    'cwlowrank': scorer_lowrank,
    'cpwlowrank': scorer_lowrank,
    'clowrank': scorer_lowrank,
    'csfull': scorer_norms,
    'wcsfull': scorer_norms,
    'gnormfull': scorer_norms,
    'unormfull': scorer_norms,
    'csrelu': scorer_nonlinear_norms,
    'cssilu': scorer_nonlinear_norms,
    'oracle': scorer_oracle,
    'oracle_out': scorer_oracle_out,
}


def parse_spec(name):
    """``topm8`` → {'kind': 'topm', 'm': 8}。数字が無ければ m は None。"""
    head = name.rstrip('0123456789')
    tail = name[len(head):]
    return {'kind': head, 'm': int(tail) if tail else None, 'name': name}


def build_scorer(name, material, fit):
    spec = parse_spec(name)
    try:
        factory = SCORERS[spec['kind']]
    except KeyError:
        raise ValueError(f'未知の score 関数 {name!r}') from None
    return factory(material, fit, spec)


# --- 採点 -----------------------------------------------------------------


@torch.no_grad()
def evaluate(material, names, chunk=2048):
    fit = material.fit_z
    built = {}
    for name in names:
        started = time.time()
        built[name] = build_scorer(name, material, fit)
        built[name] = (built[name][0], built[name][1], time.time() - started)

    topk = material.topk
    states = {name: {'exact': 0, 'overlap': 0, 'mass': 0.0, 'sq_error': 0.0}
              for name in names}
    n_tokens = 0
    total_mass = 0.0
    oracle_mass = 0.0
    output_norm = 0.0

    for x in batches(material.validation_z, chunk, material.device):
        n_tokens += x.shape[0]
        mass = true_mass(material, x)
        shared = (material.h(x, 0).abs_().sum(dim=1)
                  if material.groups[0].numel() else torch.zeros(
                      x.shape[0], device=x.device))
        total_mass += float((shared + mass.sum(dim=1)).sum(dtype=torch.float64))
        oracle_indices = mass.topk(topk, dim=1).indices
        oracle_mass += float(
            (shared + mass.gather(1, oracle_indices).sum(dim=1)).sum(
                dtype=torch.float64))
        oracle_sets = oracle_indices.sort(dim=1).values

        # 出力誤差のために expert ごとの出力を作る
        outputs = []
        for index in range(len(material.groups)):
            if not material.groups[index].numel():
                outputs.append(torch.zeros_like(x))
                continue
            _, _, down = material.expert_rows(index)
            outputs.append(material.h(x, index) @ down.T)
        dense_output = sum(outputs)
        output_norm += float(dense_output.square().sum(dtype=torch.float64))
        routed_outputs = torch.stack(outputs[1:], dim=1)

        for name in names:
            score, _, _ = built[name]
            indices = score(x).topk(topk, dim=1).indices
            states[name]['exact'] += int(
                (indices.sort(dim=1).values == oracle_sets).all(dim=1).sum())
            overlap = (indices[:, :, None] == oracle_indices[:, None, :])
            states[name]['overlap'] += int(overlap.any(dim=2).sum())
            states[name]['mass'] += float(
                (shared + mass.gather(1, indices).sum(dim=1)).sum(
                    dtype=torch.float64))
            selected = routed_outputs.gather(
                1, indices[:, :, None].expand(-1, -1, x.shape[1])).sum(dim=1)
            error = dense_output - (outputs[0] + selected)
            states[name]['sq_error'] += float(error.square().sum(dtype=torch.float64))
        del outputs, dense_output, routed_outputs, mass

    rows = {}
    for name in names:
        state = states[name]
        rows[name] = {
            'exact': state['exact'] / n_tokens,
            'recall': state['overlap'] / (n_tokens * topk),
            'mass': state['mass'] / total_mass,
            'err': math.sqrt(state['sq_error'] / output_norm),
            'macs': built[name][1],
            'build_seconds': built[name][2],
        }
    rows['_oracle_mass'] = oracle_mass / total_mass
    rows['_n_tokens'] = n_tokens
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', default=DEFAULT_DIR)
    parser.add_argument('--scorers', default='cmoe,topm4,topm16,topm64')
    parser.add_argument('--layers', default=None)
    parser.add_argument('--chunk', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    names = [field.strip() for field in args.scorers.split(',') if field.strip()]
    paths = sorted(
        os.path.join(args.dump, entry) for entry in os.listdir(args.dump)
        if entry.endswith('.pt'))
    if args.layers:
        wanted = {int(field) for field in args.layers.split(',')}
        paths = [path for path in paths
                 if int(os.path.basename(path)[5:7]) in wanted]

    device = torch.device(args.device)
    per_layer = {}
    for path in paths:
        material = Material(torch.load(path, map_location='cpu'), device)
        rows = evaluate(material, names, chunk=args.chunk)
        per_layer[material.layer] = rows
        print(f'--- 層 {material.layer} (oracle mass {rows["_oracle_mass"]:.4f}) ---')
        for name in names:
            row = rows[name]
            print(f'  {name:<14} exact={row["exact"]:.4f} recall={row["recall"]:.4f} '
                  f'mass={row["mass"]:.6f} err={row["err"]:.4f} '
                  f'MAC={row["macs"]/1e3:.0f}k ({row["build_seconds"]:.1f}s)')
        del material
        torch.cuda.empty_cache()

    print('\n=== 層平均 ===')
    baseline = {
        'mass': sum(rows['cmoe']['mass'] for rows in per_layer.values())
        / len(per_layer) if 'cmoe' in names else None}
    oracle = sum(rows['_oracle_mass'] for rows in per_layer.values()) / len(per_layer)
    print(f'{"scorer":<14} {"exact":>7} {"recall":>7} {"mass":>9} {"gap":>8} '
          f'{"err":>7} {"MAC":>8}')
    for name in names:
        values = [rows[name] for rows in per_layer.values()]
        mass = sum(row['mass'] for row in values) / len(values)
        gap = ((mass - baseline['mass']) / (oracle - baseline['mass'])
               if baseline['mass'] is not None and oracle > baseline['mass'] else float('nan'))
        print(f'{name:<14} '
              f'{sum(row["exact"] for row in values)/len(values):>7.4f} '
              f'{sum(row["recall"] for row in values)/len(values):>7.4f} '
              f'{mass:>9.6f} {gap:>7.1%} '
              f'{sum(row["err"] for row in values)/len(values):>7.4f} '
              f'{values[0]["macs"]/1e3:>7.0f}k')


if __name__ == '__main__':
    main()
