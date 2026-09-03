"""ルーターの診断: 同じ validation 軌道の上で複数のルーターを並べて測る。

測るのは4つ。

* ``router_r``  … 選ばれた expert が回収した |h| 質量の割合
* ``oracle_r``  … 真の |h| を読むオラクルの回収率（同じ分割での上限）
* Oracle との Top-K の一致（recall と完全一致率）
* 代表 score と expert 全体の質量の Pearson 相関

回収率は PPL を予測しない、というのが report/10・12・13 の一貫した結論である。
これは選択の質を見る診断であって、採否の判断は PPL で行う。

CMoE-ref の ``routerlab/methods/oracle_correlation.py`` にあった
``evaluate_routers_against_abs_oracle`` の移送。
"""

import math

import torch
import torch.nn.functional as F

from cmoe.router.methods.oracle_correlation import (activation, flatten_z,
                                                    group_indices,
                                                    pearson_from_sums,
                                                    validate_weights)


def _router_scores(router, z):
    """そのルーターが Top-K を決めるのに読んでいる score。

    代表ニューロン型（方式1〜5）は gate と classifier の積で決まるが、それ以外の
    族は自分の形を知っているので ``routing_scores`` で名乗る。相関の欄が
    「そのルーターが実際に見ている量」であることが、族をまたいでも保たれる。
    """
    scores = getattr(router, 'routing_scores', None)
    if scores is not None:
        return scores(z).float()
    if not hasattr(router, 'gate') or not hasattr(router, 'classifier'):
        raise TypeError('score を名乗らないルーターは診断にかけられない')
    return (router.classifier(z) * F.silu(router.gate(z))).abs().float()


@torch.no_grad()
def evaluate_routers_against_abs_oracle(routers, z, expert_groups, gate_weight,
                                        up_weight, chunk_size=4096):
    """固定された複数のルーターを、1本の共有 validation 軌道で比較する。"""
    validate_weights(gate_weight, up_weight)
    z = flatten_z(z, gate_weight.shape[1])
    if not routers:
        raise ValueError('ルーターが1つも渡されていない')
    n_tokens = z.shape[0]
    device = gate_weight.device
    groups = []
    tensors = []
    for group_index, group in enumerate(expert_groups):
        if group_index == 0 and not group:
            groups.append(())
            tensors.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        clean, indices = group_indices(group, gate_weight.shape[0], device)
        groups.append(clean)
        tensors.append(indices)
    routed_groups = groups[1:]
    n_routed = len(routed_groups)
    topks = {getattr(router, 'topk', None) for router in routers.values()}
    if len(topks) != 1:
        raise ValueError(f'ルーターごとに Top-K が違う: {topks}')
    topk = topks.pop()
    if not isinstance(topk, int) or not 0 < topk <= n_routed:
        raise ValueError(f'Top-K {topk} は routed expert {n_routed} 個に対して不正')

    states = {}
    for name in routers:
        states[name] = {
            'recovered': 0.0,
            'overlap': 0,
            'exact': 0,
            'n_selected': 0,
            'counts': torch.zeros(n_routed, dtype=torch.int64),
            'sum_x': torch.zeros(n_routed, dtype=torch.float64, device=device),
            'sum_x2': torch.zeros(n_routed, dtype=torch.float64, device=device),
            'sum_xy': torch.zeros(n_routed, dtype=torch.float64, device=device),
            'min_x': torch.full(
                (n_routed,), math.inf, dtype=torch.float32, device=device),
            'max_x': torch.full(
                (n_routed,), -math.inf, dtype=torch.float32, device=device),
        }
    sum_y = torch.zeros(n_routed, dtype=torch.float64, device=device)
    sum_y2 = torch.zeros_like(sum_y)
    min_y = torch.full((n_routed,), math.inf, dtype=torch.float32, device=device)
    max_y = torch.full((n_routed,), -math.inf, dtype=torch.float32, device=device)
    total = 0.0
    oracle_recovered = 0.0

    for start in range(0, n_tokens, chunk_size):
        chunk = z[start:start + chunk_size].to(device)
        masses = []
        for indices in tensors:
            if not indices.numel():
                masses.append(torch.zeros(
                    chunk.shape[0], dtype=torch.float32, device=device))
                continue
            masses.append(activation(
                chunk,
                gate_weight.index_select(0, indices),
                up_weight.index_select(0, indices),
            ).abs_().sum(dim=1, dtype=torch.float32))
        shared = masses[0]
        routed = torch.stack(masses[1:], dim=1)
        per_token_total = shared + routed.sum(dim=1)
        total += float(per_token_total.sum(dtype=torch.float64))
        oracle_indices = routed.topk(topk, dim=1).indices
        oracle_selected = routed.gather(1, oracle_indices).sum(dim=1)
        oracle_recovered += float((shared + oracle_selected).sum(dtype=torch.float64))
        sum_y.add_(routed.sum(dim=0, dtype=torch.float64))
        sum_y2.add_(routed.square().sum(dim=0, dtype=torch.float64))
        min_y.copy_(torch.minimum(min_y, routed.amin(dim=0)))
        max_y.copy_(torch.maximum(max_y, routed.amax(dim=0)))

        oracle_sets = oracle_indices.sort(dim=1).values
        oracle_mask = torch.zeros_like(routed, dtype=torch.bool)
        oracle_mask.scatter_(1, oracle_indices, True)
        for name, router in routers.items():
            weights, selection = router(chunk)
            if not torch.equal(weights, torch.ones_like(weights)):
                raise ValueError(f'{name} が expert の出力重みを変えている')
            state = states[name]
            if selection.dtype == torch.bool:
                # 可変 Top-K。集合の大きさがトークンごとに違うので、
                # 「何個走ったか」も一緒に数える。recall の分母は予算 K の
                # ままにしてある — 予算を超えて選べば上がるのが正しい
                if tuple(selection.shape) != (chunk.shape[0], n_routed):
                    raise ValueError(
                        f'{name} が返した選択の形が {tuple(selection.shape)}')
                selected = (routed * selection).sum(dim=1)
                state['exact'] += int(
                    (selection == oracle_mask).all(dim=1).sum())
                state['overlap'] += int((selection & oracle_mask).sum())
                state['counts'].add_(selection.sum(dim=0).cpu())
                state['n_selected'] += int(selection.sum())
            else:
                if tuple(selection.shape) != (chunk.shape[0], topk):
                    raise ValueError(
                        f'{name} が返した選択の形が {tuple(selection.shape)}')
                if selection.numel() and (int(selection.min()) < 0
                                          or int(selection.max()) >= n_routed):
                    raise ValueError(f'{name} が範囲外の expert を選んだ')
                selected = routed.gather(1, selection).sum(dim=1)
                sorted_indices = selection.sort(dim=1).values
                state['exact'] += int((sorted_indices == oracle_sets).all(dim=1).sum())
                overlap = (selection[:, :, None] == oracle_indices[:, None, :])
                state['overlap'] += int(overlap.any(dim=2).sum())
                state['counts'].add_(
                    torch.bincount(selection.flatten().cpu(), minlength=n_routed))
                state['n_selected'] += int(selection.numel())
            state['recovered'] += float((shared + selected).sum(dtype=torch.float64))
            scores = _router_scores(router, chunk)
            state['sum_x'].add_(scores.sum(dim=0, dtype=torch.float64))
            state['sum_x2'].add_(scores.square().sum(dim=0, dtype=torch.float64))
            state['sum_xy'].add_((scores * routed).sum(dim=0, dtype=torch.float64))
            state['min_x'].copy_(torch.minimum(state['min_x'], scores.amin(dim=0)))
            state['max_x'].copy_(torch.maximum(state['max_x'], scores.amax(dim=0)))

    if not total > 0:
        raise ValueError(f'validation の活性量が {total}')
    result = {}
    for name, state in states.items():
        correlations = []
        for index in range(n_routed):
            try:
                values, constant = pearson_from_sums(
                    n_tokens,
                    state['sum_x'][index:index + 1],
                    state['sum_x2'][index:index + 1],
                    state['sum_xy'][index:index + 1],
                    sum_y[index], sum_y2[index],
                    state['min_x'][index:index + 1],
                    state['max_x'][index:index + 1],
                    float(min_y[index]), float(max_y[index]),
                )
            except ValueError:
                correlations.append(None)
            else:
                correlations.append(None if bool(constant[0]) else float(values[0]))
        result[name] = {
            'router_r': state['recovered'] / total,
            'oracle_r': oracle_recovered / total,
            'oracle_mean_recall': state['overlap'] / (n_tokens * topk),
            'oracle_exact_set_rate': state['exact'] / n_tokens,
            # 可変 Top-K のルーターでは K と一致しない。予算が守られている
            # かどうかは、比較が成立する条件そのものなので必ず残す
            'mean_selected': state['n_selected'] / n_tokens,
            'n_selected_per_expert': state['counts'].tolist(),
            'representative_mass_correlations': correlations,
            'mean_representative_mass_correlation': (
                sum(value for value in correlations if value is not None)
                / sum(value is not None for value in correlations)
                if any(value is not None for value in correlations) else None
            ),
            'n_tokens': n_tokens,
        }
    return result


def gap_recovered(baseline_r, router_r, oracle_r):
    """現行との差を、オラクルまでの差で割ったもの（gap 回収率）。"""
    denominator = oracle_r - baseline_r
    if abs(denominator) <= 1e-12:
        return None
    return (router_r - baseline_r) / denominator
