"""[軸3] ニューロン分割の境界。

``Partition`` は「どのニューロンがどの expert に属するか」だけを持ち、重みの
コピーも Router も持たない。expert の重みは ``build_experts`` が Partition と
dense FFN から作る（分割を測るだけの用途では重みを作らずに済ませたいため）。

設計計画では Partition から x 依存を外すことを掲げているが、現行 CMoE の分割
規則は shared の取り分を先に抜いてから残りをクラスタリングするので、分割は x に
依存する。x 非依存の分割は分割アルゴリズム側の変更であり、この境界の形とは
別の話として残してある。
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from cmoe.moe.modules import SubMLP


@dataclass(frozen=True)
class Partition:
    """一層分の分割。

    expert_groups[0] は shared expert のニューロン（x=0 なら空）、
    expert_groups[1:] が routed expert。
    representative_indices は routed expert ごとの代表ニューロン1個で、
    現行ルーターはこの行から作られる。
    """

    n_experts: int
    n_shared: int
    expert_groups: tuple
    representative_indices: tuple

    @property
    def n_routed(self):
        return self.n_experts - self.n_shared

    @property
    def shared_group(self):
        return self.expert_groups[0]

    @property
    def routed_groups(self):
        return self.expert_groups[1:]

    def check(self, intermediate_size):
        covered = sum(len(group) for group in self.expert_groups)
        if covered != intermediate_size:
            raise ValueError(
                f'分割が {covered} ニューロンしか覆っていない '
                f'(FFN は {intermediate_size})')
        if len(self.routed_groups) != self.n_routed:
            raise ValueError(
                f'routed expert が {len(self.routed_groups)} 個、'
                f'{self.n_routed} 個のはず')
        if len(self.representative_indices) != self.n_routed:
            raise ValueError(
                f'代表が {len(self.representative_indices)} 個、'
                f'{self.n_routed} 個のはず')
        for expert, (rep, group) in enumerate(
                zip(self.representative_indices, self.routed_groups)):
            if rep not in group:
                raise ValueError(f'expert {expert} の代表が自分の外にいる')
        return self


class CarveMethod(Protocol):
    """dense FFN と活性統計から分割を作る。"""

    name: str

    def carve(self, dense, rates, markers, n_shared, z=None,
              layer=None) -> Partition:
        ...


@torch.no_grad()
def build_experts(dense, partition):
    """Partition の各グループへ、dense FFN の該当行をコピーした SubMLP を作る。

    戻り値は expert_groups と同じ並び（先頭が shared）。
    """
    experts = []
    for indices in partition.expert_groups:
        expert = SubMLP(dense.hidden_size, len(indices)).to('cpu')
        expert.gate_proj.weight.data = dense.gate_proj.weight.data[indices, :]
        expert.up_proj.weight.data = dense.up_proj.weight.data[indices, :]
        expert.down_proj.weight.data = dense.down_proj.weight.data[:, indices]
        experts.append(expert)
    return experts
