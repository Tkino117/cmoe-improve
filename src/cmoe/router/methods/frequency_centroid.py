"""方式2: 活性頻度で重み付けした重心に最も近いニューロンを代表にする。

現行 CMoE（方式1）は活性 marker の単純重心に最も近いニューロンを選ぶ。この方式は
marker を各ニューロンの活性頻度で重み付けしてから重心を取る — よく発火する
メンバーに寄った重心である。推論の形（1 expert・1代表）は変えない。

CMoE-ref の ``routerlab/methods/frequency_centroid.py`` の移送。
"""

import copy

import torch
import torch.nn.functional as F


def _validate_statistics(rates, markers):
    if rates is None or markers is None:
        raise ValueError('freq_centroid は活性統計を要求する')
    if rates.dim() != 1 or markers.dim() != 2:
        raise ValueError(
            f'rates は [neurons]、markers は [tokens, neurons] のはず。'
            f'{tuple(rates.shape)} と {tuple(markers.shape)} を受け取った')
    if markers.shape[1] != rates.shape[0]:
        raise ValueError(
            f'markers は {markers.shape[1]} ニューロン、rates は {rates.shape[0]}')
    if not bool(torch.isfinite(rates).all()) or bool((rates < 0).any()):
        raise ValueError('活性頻度は有限かつ非負のはず')
    if not bool(torch.isfinite(markers).all()):
        raise ValueError('活性 marker は有限のはず')
    if bool(((markers != 0) & (markers != 1)).any()):
        raise ValueError('活性 marker は 0/1 のはず')


@torch.no_grad()
def frequency_weighted_representatives(routed_groups, rates, markers, chunk_size=256):
    """頻度重み付き重心に L1 距離で最も近い実在ニューロンを expert ごとに選ぶ。

    候補ニューロンで分割して進めるので、7B でも全 marker の float32 コピーを
    一度に作らない。完全に同点なら元のニューロン番号が小さい方。
    """
    _validate_statistics(rates, markers)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(f'chunk_size は正の整数のはず (受け取った値: {chunk_size!r})')

    rates = rates.detach().to(device='cpu', dtype=torch.float32)
    markers = markers.detach().to(device='cpu')
    n_neurons = rates.shape[0]
    representatives = []

    for expert_index, group in enumerate(routed_groups):
        group = tuple(int(index) for index in group)
        if not group:
            raise ValueError(f'routed expert {expert_index} が空')
        if len(set(group)) != len(group):
            raise ValueError(f'routed expert {expert_index} にニューロンの重複がある')
        if min(group) < 0 or max(group) >= n_neurons:
            raise ValueError(
                f'routed expert {expert_index} に 0..{n_neurons - 1} の外の番号がある')

        group_rates = rates[list(group)]
        denominator = float(group_rates.sum())
        if not denominator > 0:
            # このグループの marker はすべて零ベクトルということになる。重み付き
            # 重心は定義できないが、単純重心の規則では全員が同点になる。方式1 に
            # 帰着させるため、同点の解き方（最小番号）をそのまま使う。
            representatives.append(min(group))
            continue

        centroid = torch.zeros(markers.shape[0], dtype=torch.float32)
        for start in range(0, len(group), chunk_size):
            stop = min(start + chunk_size, len(group))
            indices = list(group[start:stop])
            block = markers[:, indices].to(torch.float32)
            centroid.add_(block.mv(group_rates[start:stop]))
        centroid.div_(denominator)

        best_distance = None
        best_neuron = None
        for start in range(0, len(group), chunk_size):
            stop = min(start + chunk_size, len(group))
            indices = list(group[start:stop])
            block = markers[:, indices].to(torch.float32)
            distances = (block - centroid[:, None]).abs_().sum(dim=0)
            for neuron, distance in zip(indices, distances.tolist()):
                if (best_distance is None or distance < best_distance
                        or (distance == best_distance and neuron < best_neuron)):
                    best_distance = distance
                    best_neuron = neuron
        representatives.append(best_neuron)

    return tuple(representatives)


def install_representatives(baseline, context, representatives, selection):
    """選んだ代表の行をルーターに書き込む。方式2〜4 で共通。

    gate には gate の行、classifier には up の行が入る。正規化の有無は
    context が持つ（分割の側の設定と必ず一致する）。
    """
    router = copy.deepcopy(baseline)
    indices = list(representatives)
    gate = context.gate_weight[indices]
    up = context.up_weight[indices]
    if context.router_normalized:
        gate = F.normalize(gate, p=2, dim=1)
        up = F.normalize(up, p=2, dim=1)
    router.gate.weight.data.copy_(gate)
    router.classifier.weight.data.copy_(up)
    router.representative_indices = tuple(representatives)
    router.selection = selection
    return router


class FrequencyCentroidMethod:
    name = 'freq_centroid'
    training_free = True
    requires_source_weights = True
    requires_profiling_stats = True
    diagnostic_only = False

    def build(self, context, baseline):
        if context.gate_weight is None or context.up_weight is None:
            raise ValueError('freq_centroid は dense の gate/up 重みを要求する')
        representatives = frequency_weighted_representatives(
            context.routed_groups,
            context.activation_rates,
            context.activation_markers,
        )
        return install_representatives(baseline, context, representatives, {
            'rule': 'activation-frequency-weighted-centroid-l1',
            'source': 'partition-carving-markers',
        })
