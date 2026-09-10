"""分割方式名の解決。"""

from cmoe.carve.cmoe import CMoECarver
from cmoe.carve.llama_moe import LlamaMoEV2Carver, RandomSplitCarver
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
    # [対照] 先行研究の分割規則。**どちらも提案手法ではない**（report/26）
    'llama_moe_random': RandomSplitCarver,
    'llama_moe_v2': LlamaMoEV2Carver,
}


def create_carver(name, n_experts, k_act=None, seed=None, scores=None):
    """``seed`` は分割そのものが確率的な方式（``llama_moe_random``）だけが読む。
    ``scores`` は層ごとの統計を外から受け取る方式（``llama_moe_v2``）だけが読む。
    """
    try:
        carver = CARVERS[name]
    except KeyError:
        raise ValueError(
            f'未知の分割方式 {name!r}。{sorted(CARVERS)} から選ぶ') from None
    if carver is RandomSplitCarver:
        built = carver(n_experts, seed=seed or 0)
    elif carver is LlamaMoEV2Carver:
        if scores is None:
            raise ValueError(
                'llama_moe_v2 は層ごとの重要度が要る。'
                'experiments/28_llama_moe/probe.py の出力を --carve-scores で渡す')
        built = carver(n_experts, scores=scores)
    else:
        built = carver(n_experts)
    # 大きさを見る方式は、プロファイルと同じ上位 k を使わないと、印の位置が
    # markers とずれる
    if k_act is not None:
        built.k_act = k_act
    return built
