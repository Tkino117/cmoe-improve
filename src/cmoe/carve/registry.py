"""分割方式名の解決。"""

from cmoe.carve.cmoe import CMoECarver
from cmoe.carve.variants import (IteratedCMoECarver, MassSharedCarver,
                                 WeightedMarkerCarver)

CARVERS = {
    # 現行 CMoE。すべての実験の対照
    'cmoe': CMoECarver,
    # 反復を1→5
    'cmoe_iter5': IteratedCMoECarver,
    # クラスタリングの特徴を 0/1 から |h| へ
    'cmoe_weighted': WeightedMarkerCarver,
    # shared の選び方を頻度から質量へ
    'cmoe_mass': MassSharedCarver,
}


def create_carver(name, n_experts, k_act=None):
    try:
        carver = CARVERS[name]
    except KeyError:
        raise ValueError(
            f'未知の分割方式 {name!r}。{sorted(CARVERS)} から選ぶ') from None
    built = carver(n_experts)
    # 大きさを見る方式は、プロファイルと同じ上位 k を使わないと、印の位置が
    # markers とずれる
    if k_act is not None:
        built.k_act = k_act
    return built
