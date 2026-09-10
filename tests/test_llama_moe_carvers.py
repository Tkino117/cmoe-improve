"""LLaMA-MoE / LLaMA-MoE-v2 の分割規則（対照）の動作確認。

規則の中身は公式実装の移送なので、ここが見るのは
「規則どおりの分割が出ているか」と「土俵の前提（等サイズ・非重複・全被覆）を
壊していないか」である。**手法の質は見ない。**
"""

import torch
import pytest

from cmoe.carve.llama_moe import (LlamaMoEV2Carver, RandomSplitCarver,
                                  _greedy_assign, _residual_neurons)
from cmoe.carve.registry import create_carver
from cmoe.adapters.base import DenseFFN

N_EXPERTS, INTER, HIDDEN = 4, 32, 8


@pytest.fixture
def dense():
    import torch.nn as nn

    torch.manual_seed(0)
    return DenseFFN(hidden_size=HIDDEN, intermediate_size=INTER,
                    gate_proj=nn.Linear(HIDDEN, INTER, bias=False),
                    up_proj=nn.Linear(HIDDEN, INTER, bias=False),
                    down_proj=nn.Linear(INTER, HIDDEN, bias=False),
                    act_fn=nn.SiLU())


@pytest.fixture
def stats():
    torch.manual_seed(1)
    rates = torch.rand(INTER)
    markers = torch.rand(6, INTER)
    return rates, markers


def assert_valid(partition, n_shared):
    """等サイズ・非重複・全被覆。この土俵の前提そのもの。"""
    size = INTER // N_EXPERTS
    seen = set()
    for group in partition.expert_groups:
        assert not (seen & set(group)), '同じニューロンが2つの expert にいる'
        seen |= set(group)
    assert len(seen) == INTER, 'ニューロンを覆い尽くしていない'
    assert len(partition.shared_group) == n_shared * size
    for group in partition.routed_groups:
        assert len(group) == size


# -- v1: Random -----------------------------------------------------------

@pytest.mark.parametrize('n_shared', [0, 1, 2])
def test_random_split_is_equal_sized_and_disjoint(dense, stats, n_shared):
    rates, markers = stats
    carver = RandomSplitCarver(N_EXPERTS, seed=0)
    assert_valid(carver.carve(dense, rates, markers, n_shared, layer=0), n_shared)


def test_random_split_reads_no_calibration(dense, stats):
    """校正を替えても分割が変わらないこと。原典は重みもデータも見ない。"""
    rates, markers = stats
    carver = RandomSplitCarver(N_EXPERTS, seed=0)
    first = carver.carve(dense, rates, markers, 1, layer=3)
    second = carver.carve(dense, torch.rand(INTER), torch.rand(6, INTER), 1,
                          layer=3)
    assert first.expert_groups == second.expert_groups


def test_random_split_differs_by_layer_and_seed(dense, stats):
    """層ごと・seed ごとに別の割り当てになること。

    層をまたいで同じ並びを使うと「ランダム分割」ではなく「1つの固定分割」を
    測ることになる。seed で変わらないなら、seed を振る意味が消える。
    """
    rates, markers = stats
    carver = RandomSplitCarver(N_EXPERTS, seed=0)
    layer0 = carver.carve(dense, rates, markers, 1, layer=0).expert_groups
    layer1 = carver.carve(dense, rates, markers, 1, layer=1).expert_groups
    other = RandomSplitCarver(N_EXPERTS, seed=1).carve(
        dense, rates, markers, 1, layer=0).expert_groups
    assert layer0 != layer1
    assert layer0 != other
    # 同じ seed・同じ層なら同じ
    assert layer0 == RandomSplitCarver(N_EXPERTS, seed=0).carve(
        dense, rates, markers, 1, layer=0).expert_groups


# -- v2: Gradient + Residual ----------------------------------------------

def test_residual_takes_the_neurons_that_most_clusters_want():
    """residual は「多くのクラスタが上位に挙げたニューロン」から埋まる。

    3クラスタの上位2件が {0,1} {0,2} {0,3} なら、3クラスタ全部が挙げた
    ニューロン 0 が最初に residual へ入る。
    """
    order = [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5]]
    picked = _residual_neurons(order, need=1, size=2, n_routed=3)
    assert picked == [0]
    # 2つ要るなら、0 を除いたあとの上位2件から選び直す
    picked = _residual_neurons(order, need=2, size=2, n_routed=3)
    assert picked[0] == 0 and len(picked) == 2 and len(set(picked)) == 2


def test_greedy_assign_takes_the_globally_largest_score():
    """貪欲選択が見るのは順位ではなくスコアそのもの。

    クラスタ1 の最上位 (0.9) がクラスタ0 の最上位 (0.5) より大きいので、
    先に取られるのはクラスタ1 の側である。
    """
    table = torch.tensor([[0.5, 0.4, 0.1, 0.0],
                          [0.9, 0.0, 0.8, 0.1]])
    order = [torch.argsort(row, descending=True).tolist() for row in table]
    groups = _greedy_assign(table, order, residual=[], n_routed=2, size=2)
    assert 0 in groups[1] and 2 in groups[1]
    assert sorted(groups[0]) == [1, 3]


@pytest.mark.parametrize('n_shared', [0, 1, 2])
def test_v2_split_is_equal_sized_and_disjoint(dense, stats, n_shared):
    rates, markers = stats
    torch.manual_seed(2)
    scores = {0: torch.rand(N_EXPERTS - n_shared, INTER)}
    carver = LlamaMoEV2Carver(N_EXPERTS, scores=scores)
    assert_valid(carver.carve(dense, rates, markers, n_shared, layer=0), n_shared)


def test_v2_refuses_without_a_layer(dense, stats):
    """探索の経路からは使えない。黙って別の層の統計を使わないための断り。"""
    rates, markers = stats
    carver = LlamaMoEV2Carver(N_EXPERTS, scores={0: torch.rand(3, INTER)})
    with pytest.raises(ValueError, match='層番号'):
        carver.carve(dense, rates, markers, 1)


def test_v2_refuses_a_score_table_of_the_wrong_shape(dense, stats):
    """クラスタ数 = routed 数。x を変えたらプローブを取り直す必要がある。"""
    rates, markers = stats
    carver = LlamaMoEV2Carver(N_EXPERTS, scores={0: torch.rand(2, INTER)})
    with pytest.raises(ValueError, match='routed'):
        carver.carve(dense, rates, markers, 1, layer=0)


def test_registry_refuses_v2_without_scores():
    with pytest.raises(ValueError, match='重要度'):
        create_carver('llama_moe_v2', N_EXPERTS)


def test_registry_passes_the_seed_to_the_random_split(dense, stats):
    rates, markers = stats
    left = create_carver('llama_moe_random', N_EXPERTS, seed=0)
    right = create_carver('llama_moe_random', N_EXPERTS, seed=1)
    assert (left.carve(dense, rates, markers, 1, layer=0).expert_groups
            != right.carve(dense, rates, markers, 1, layer=0).expert_groups)
