"""現行 CMoE の分割規則。

CMoE-ref の ``CMoE_utils.construct_experts_k_means`` の移送。手順は3段:

1. 活性頻度の上位から shared expert の取り分（neurons_per_expert * x 個）を抜く
2. 残りを共活性パターン(markers)の L1 距離で、大きさを揃えた k 個に割る
   （lap.lapjv による等サイズ割り当て。反復は1回）
3. 各クラスタの重心に最も近いニューロンを代表とする

数値を既存の測定と一致させるため、演算の順序と精度はそのままにしてある。
重みのコピーはここでは行わず、``carve.base.build_experts`` が担う。
"""

import lap
import numpy as np
import torch

from cmoe.carve.base import Partition


class CMoECarver:
    """現行 CMoE の分割。すべての実験の対照になる。"""

    name = 'cmoe'

    def __init__(self, n_experts):
        if n_experts < 1:
            raise ValueError(f'n_experts は 1 以上 (受け取った値: {n_experts})')
        self.n_experts = n_experts

    @torch.no_grad()
    def carve(self, dense, rates, markers, n_shared):
        if not 0 <= n_shared < self.n_experts:
            raise ValueError(
                f'n_shared={n_shared} は 0..{self.n_experts - 1} の外')
        if rates.shape[0] != dense.intermediate_size:
            raise ValueError(
                f'統計は {rates.shape[0]} ニューロン分、FFN は '
                f'{dense.intermediate_size} — 統計と層が対応していない')
        if markers.shape[-1] != rates.shape[0]:
            raise ValueError(
                f'markers は {markers.shape[-1]} 、rates は {rates.shape[0]}')
        if rates.shape[0] % self.n_experts:
            # 元実装は neurons_per_expert を切り捨てるので、余りのニューロンは
            # どの expert にも入らずモデルから消える
            raise ValueError(
                f'{rates.shape[0]} ニューロンは {self.n_experts} で割り切れない; '
                '余りが黙って捨てられる')

        groups, representatives = _kmeans_groups(
            rates, markers, self.n_experts, n_shared)
        partition = Partition(
            n_experts=self.n_experts,
            n_shared=n_shared,
            expert_groups=tuple(tuple(group) for group in groups),
            representative_indices=tuple(representatives),
        )
        return partition.check(dense.intermediate_size)


@torch.no_grad()
def _kmeans_groups(activation_rates, activation_markers, num_experts,
                   num_shared_experts):
    hidden_size = activation_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    remaining_indices = set(range(hidden_size))
    markers = activation_markers.float()

    _, top_indices = torch.topk(activation_rates, neurons_per_expert * num_shared_experts)
    shared_expert_indices = top_indices.tolist()
    expert_groups.append(shared_expert_indices)
    remaining_indices -= set(shared_expert_indices)
    remaining_indices_list = list(remaining_indices)

    n_samples = len(remaining_indices_list)
    k = num_experts - num_shared_experts
    cluster_size = neurons_per_expert

    remaining_rates = activation_rates[remaining_indices_list]
    if num_shared_experts == 0:
        # x>=1 の場合と種の候補プールを揃える（順位 neurons_per_expert+1 以降）
        _, top_indices = torch.topk(remaining_rates, neurons_per_expert + k)
        top_indices = top_indices[neurons_per_expert:]
    else:
        _, top_indices = torch.topk(remaining_rates, k)
    selected_index = [remaining_indices_list[idx] for idx in top_indices]

    centroids = markers[:, selected_index].clone()

    max_iters = 1
    prev_assignments = None

    for _ in range(max_iters):
        distances = torch.cdist(markers[:, remaining_indices_list].T, centroids.T, p=1)
        distances_np = distances.numpy()
        repeated_distances = np.zeros((n_samples, n_samples))
        for i in range(k):
            repeated_distances[:, i * cluster_size:(i + 1) * cluster_size] = distances_np[:, i:i + 1]

        row_ind, col_ind = lap.lapjv(repeated_distances)[0:2]
        assignments = torch.tensor(col_ind // cluster_size)

        if prev_assignments is not None and torch.all(assignments == prev_assignments):
            break

        prev_assignments = assignments.clone()

        for i in range(k):
            cluster_points = markers[:, remaining_indices_list][:, assignments == i]
            if cluster_points.size(1) > 0:
                centroids[:, i] = cluster_points.mean(dim=1)

    representative_indices = []
    for i in range(k):
        cluster_mask = assignments == i
        cluster_points = markers[:, remaining_indices_list][:, cluster_mask]
        cluster_indices = torch.where(cluster_mask)[0]

        distances = torch.cdist(centroids[:, i:i + 1].T, cluster_points.T, p=1)

        closest_idx_in_cluster = torch.argmin(distances)
        original_idx = cluster_indices[closest_idx_in_cluster]
        representative_indices.append(remaining_indices_list[original_idx])
        expert_groups.append([remaining_indices_list[idx] for idx in cluster_indices])

    return expert_groups, representative_indices
