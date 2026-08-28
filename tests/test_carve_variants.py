"""分割規則の派生3つ。CPU、数秒。

見るのは2点だけである。**現行の分割が1ビットも動いていないこと**（対照が
動いたら比較にならない）と、**各派生が意図した1箇所だけを変えていること**。

派生の良し悪しはここでは測らない。それは回収率と PPL とベンチの仕事である。
"""

import pytest
import torch

from cmoe.carve.cmoe import CMoECarver
from cmoe.carve.profile import (analyze_activations, hidden_activations,
                                marker_weights, neuron_mass, profile_layer)
from cmoe.carve.registry import CARVERS, create_carver

N_EXPERTS, INTER, HIDDEN, TOKENS = 4, 32, 8, 24


@pytest.fixture
def layer():
    """小さな FFN と、その入力。"""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP

    config = LlamaConfig(hidden_size=HIDDEN, intermediate_size=INTER)
    torch.manual_seed(0)
    dense = LlamaMLP(config).eval()
    z = torch.randn(2, TOKENS // 2, HIDDEN, generator=torch.Generator().manual_seed(1))
    return dense, z


def profile(dense, z, k_act=4):
    return profile_layer(dense, z, k_act=k_act)


def test_every_registered_carver_makes_a_valid_partition(layer):
    dense, z = layer
    rates, markers = profile(dense, z)
    for name in CARVERS:
        carver = create_carver(name, N_EXPERTS, k_act=4)
        partition = carver.carve(dense, rates, markers, 2, z=z)
        assert partition.n_shared == 2
        assert sum(len(group) for group in partition.expert_groups) == INTER


def test_the_current_rule_is_untouched_by_the_refactor(layer):
    """既定の経路は z を読まない。読まずに同じ分割が出ることを確かめる。"""
    dense, z = layer
    rates, markers = profile(dense, z)
    carver = CMoECarver(N_EXPERTS)
    without = carver.carve(dense, rates, markers, 2)
    with_z = carver.carve(dense, rates, markers, 2, z=z)
    assert without.expert_groups == with_z.expert_groups
    assert without.representative_indices == with_z.representative_indices


def test_more_iterations_only_changes_the_iteration_count(layer):
    dense, z = layer
    rates, markers = profile(dense, z)
    base = create_carver('cmoe', N_EXPERTS)
    iterated = create_carver('cmoe_iter5', N_EXPERTS)
    assert base.max_iters == 1 and iterated.max_iters == 5
    # shared は頻度で切るので、反復を増やしても shared 群は変わらない
    assert (base.carve(dense, rates, markers, 2).shared_group
            == iterated.carve(dense, rates, markers, 2).shared_group)


def test_weighted_markers_keep_the_positions_and_change_the_values(layer):
    dense, z = layer
    _, markers = profile(dense, z)
    weights = marker_weights(dense, z, k_act=4)
    assert weights.shape == markers.shape
    assert torch.equal((weights > 0).float(), markers)
    assert not torch.equal(weights, markers)


def test_the_weighted_carver_uses_those_values(layer):
    dense, z = layer
    rates, markers = profile(dense, z)
    carver = create_carver('cmoe_weighted', N_EXPERTS, k_act=4)
    features = carver._features(dense, rates, markers, z)
    assert torch.equal((features > 0).float(), markers)
    # shared の選び方は変えていない
    assert torch.equal(carver._shared_scores(dense, rates, markers, z), rates)


def test_neuron_mass_is_the_true_activation_summed_over_tokens(layer):
    dense, z = layer
    mass = neuron_mass(dense, z)
    h = hidden_activations(dense, z, normalize=False)
    expected = h.reshape(-1, INTER).abs().to(torch.float32).sum(dim=0)
    assert torch.allclose(mass, expected, atol=1e-5)


def test_the_mass_carver_picks_shared_by_mass_not_by_rate(layer):
    """稀にしか上位に来ないが来たときに大きいニューロンを作って、
    現行では shared に入らず、質量で切ると入ることを見る。"""
    dense, z = layer
    rates, markers = profile(dense, z)
    carver = create_carver('cmoe_mass', N_EXPERTS, k_act=4)
    scores = carver._shared_scores(dense, rates, markers, z)
    assert scores.shape == rates.shape
    assert not torch.equal(scores, rates)
    # 特徴（クラスタリング）は現行のまま
    assert torch.equal(carver._features(dense, rates, markers, z), markers)


def test_the_carvers_that_need_z_say_so_when_it_is_missing(layer):
    dense, z = layer
    rates, markers = profile(dense, z)
    for name in ('cmoe_weighted', 'cmoe_mass'):
        with pytest.raises(ValueError, match='z'):
            create_carver(name, N_EXPERTS).carve(dense, rates, markers, 2)


def test_k_act_reaches_the_carver(layer):
    assert create_carver('cmoe_weighted', N_EXPERTS, k_act=7).k_act == 7
    assert create_carver('cmoe', N_EXPERTS).k_act == 10
