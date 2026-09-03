"""report/22 の対照3本（順序対照・LExI・OWL）の、CPU で閉じる部分。

GPU が要るのは統計を作るところだけで、そこから配分に落とす写像はすべてここで
検査できる。EW-rule と同じ切り方である。
"""

import pytest
import torch

from cmoe.alloc import lexi, order, owl, score_rule


# -- 順序対照 -------------------------------------------------------------

def test_reverse_keeps_the_multiset_and_flips_the_order():
    values = (4, 6, 6, 5, 2)
    flipped = order.reverse(values)
    assert flipped == (2, 5, 6, 6, 4)
    assert sorted(flipped) == sorted(values)


def test_shuffle_is_reproducible_and_avoids_identity_and_reverse():
    values = (4, 6, 6, 5, 2, 0, 3, 1)
    first = order.shuffle(values, seed=7)
    assert order.shuffle(values, seed=7) == first
    assert sorted(first) == sorted(values)
    assert first != values
    assert first != tuple(reversed(values))


def test_shuffle_with_a_different_seed_gives_a_different_draw():
    values = tuple(range(7)) * 2
    assert order.shuffle(values, seed=1) != order.shuffle(values, seed=2)


@pytest.mark.parametrize('call', [order.reverse,
                                  lambda v: order.shuffle(v, seed=0)])
def test_a_uniform_allocation_is_rejected_as_an_order_control(call):
    # 一様配分はどう並べ替えても同じ。潰れた対照をベンチに掛けさせない
    with pytest.raises(ValueError, match='順序対照にならない'):
        call((3, 3, 3, 3))


def test_displacement_counts_how_far_the_permutation_moved_things():
    detail = order.displacement((0, 1, 2, 3), (3, 2, 1, 0))
    assert detail['n_changed'] == 4
    assert detail['max_abs_shift'] == 3
    assert detail['mean_abs_shift'] == pytest.approx(2.0)


def test_displacement_refuses_a_pair_that_is_not_a_permutation():
    with pytest.raises(ValueError, match='多重集合が違う'):
        order.displacement((1, 2, 3), (1, 2, 4))


# -- 規則型の共通の器 -----------------------------------------------------

def test_the_rule_reproduces_the_expert_weaver_map_when_the_score_is_a_ratio():
    from cmoe.alloc.ew_rule import ew_allocation

    # CV の作り方は違っても、r が同じなら x も同じでなければならない。
    # ここは r を直接与えて、写像だけが一致することを見る
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    cvs = [torch.full((100,), 2.0) for _ in ratios]
    for cv, ratio in zip(cvs, ratios):
        cv[int(ratio * 100):] = 0.0          # CV > τ=0.6 の割合が ratio になる
    expected, _ = ew_allocation(cvs, n_experts=8, n_active=6)
    values, _ = score_rule.rule_allocation(
        ratios, n_experts=8, n_active=6, alpha_min=0.2, alpha_max=0.7,
        direction='desc', normalize='none')
    assert values == expected


def test_desc_gives_less_shared_to_the_layer_with_the_larger_score():
    values, _ = score_rule.rule_allocation(
        [0.1, 0.9], n_experts=8, n_active=6, alpha_min=0.0, alpha_max=1.0)
    assert values[0] > values[1]


def test_asc_flips_the_direction():
    ascending, _ = score_rule.rule_allocation(
        [0.1, 0.9], n_experts=8, n_active=6, alpha_min=0.0, alpha_max=1.0,
        direction='asc')
    descending, _ = score_rule.rule_allocation(
        [0.1, 0.9], n_experts=8, n_active=6, alpha_min=0.0, alpha_max=1.0,
        direction='desc')
    assert ascending == list(reversed(descending))


def test_minmax_rescues_a_score_that_would_otherwise_collapse():
    # OWL の外れ値比率は 1e-3 の桁。生のまま入れると α が α_max に張り付く
    tiny = [0.0011, 0.0019, 0.0032]
    collapsed, detail = score_rule.rule_allocation(
        tiny, 8, 6, 0.2, 0.7, normalize='none')
    assert detail['degenerate'] and len(set(collapsed)) == 1
    spread, detail = score_rule.rule_allocation(tiny, 8, 6, 0.2, 0.7)
    assert not detail['degenerate'] and len(set(spread)) > 1


def test_minmax_reports_degenerate_when_every_layer_is_the_same():
    values, detail = score_rule.rule_allocation([0.4] * 5, 8, 6, 0.2, 0.7)
    assert detail['degenerate'] and len(set(values)) == 1


def test_clipping_is_counted_so_it_cannot_be_read_past():
    # α_max=1.0 は x=8 を出すが、A=4 では 4 に切られる
    values, detail = score_rule.rule_allocation(
        [0.0, 1.0], n_experts=8, n_active=4, alpha_min=0.0, alpha_max=1.0)
    assert values == [4, 0]
    assert detail['x_before_clip'] == [8, 0]
    assert detail['n_clipped'] == 1


def test_every_value_stays_inside_the_budget():
    from cmoe.alloc.base import Allocation

    scores = [0.0, 0.3, 0.55, 0.8, 1.0]
    for n_active in (4, 6):
        values, _ = score_rule.rule_allocation(
            scores, 8, n_active, 0.0, 1.0)
        Allocation(tuple(values), n_active_total=n_active)   # 例外が出ない


@pytest.mark.parametrize('bad', [('sideways', 'minmax'), ('desc', 'zscore')])
def test_unknown_knobs_are_refused(bad):
    direction, normalize = bad
    with pytest.raises(ValueError):
        score_rule.rule_allocation([0.0, 1.0], 8, 6, 0.2, 0.7,
                                   direction=direction, normalize=normalize)


def test_alpha_range_must_be_ordered():
    with pytest.raises(ValueError, match='α は'):
        score_rule.rule_allocation([0.0, 1.0], 8, 6, 0.8, 0.3)


# -- LExI -----------------------------------------------------------------

def test_argmin_picks_the_lowest_local_error_per_layer():
    table = [{0: 0.5, 1: 0.2, 2: 0.9},
             {0: 0.1, 1: 0.4, 2: 0.3}]
    values, detail = lexi.argmin_allocation(table, n_active=2)
    assert values == [1, 0]
    assert detail['sum_local_error'] == pytest.approx(0.3)
    assert detail['margin_to_runner_up'][0] == pytest.approx(0.3)


def test_ties_are_broken_by_the_recorded_preference():
    table = [{0: 0.5, 1: 0.5, 2: 0.9}]
    low, _ = lexi.argmin_allocation(table, n_active=2, prefer='low')
    high, _ = lexi.argmin_allocation(table, n_active=2, prefer='high')
    assert low == [0] and high == [1]


def test_a_table_that_always_prefers_no_routing_is_reported_as_such():
    table = [{0: 0.9, 1: 0.5, 2: 0.1}] * 4
    values, detail = lexi.argmin_allocation(table, n_active=2)
    assert values == [2, 2, 2, 2]
    assert detail['n_routing_layers'] == 0
    assert detail['n_no_routing_layers'] == 4
    assert detail['degenerate']


def test_a_candidate_outside_the_budget_is_refused():
    with pytest.raises(ValueError, match='0..2 の外'):
        lexi.argmin_allocation([{0: 0.1, 3: 0.2}], n_active=2)


# -- OWL ------------------------------------------------------------------

def test_input_norms_match_a_plain_column_norm():
    x = torch.randn(3, 5, 7)
    got = owl.input_norms(x)
    expected = x.reshape(-1, 7).float().pow(2).sum(dim=0).sqrt()
    assert torch.allclose(got, expected, atol=1e-5)


def test_input_norms_do_not_depend_on_the_chunking():
    x = torch.randn(4, 6, 9)
    assert torch.allclose(owl.input_norms(x), owl.input_norms(x, batch_chunk=2),
                          atol=1e-6)


class _TinyFFN:
    def __init__(self, hidden, inner):
        torch.manual_seed(0)
        self.gate_proj = torch.nn.Linear(hidden, inner, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inner, bias=False)
        self.down_proj = torch.nn.Linear(inner, hidden, bias=False)


def test_outlier_ratio_is_between_zero_and_one_and_falls_as_m_grows():
    dense = _TinyFFN(16, 32)
    z = torch.randn(2, 12, 16)
    h = torch.randn(2, 12, 32)
    ratios, detail = owl.layer_outlier_ratios(dense, z, h)
    assert detail['n_elements'] == 16 * 32 * 3
    assert all(0.0 <= r <= 1.0 for r in ratios.values())
    ordered = [ratios[m] for m in sorted(ratios)]
    assert ordered == sorted(ordered, reverse=True)


def test_outlier_ratio_ignores_the_chunking_of_the_activations():
    dense = _TinyFFN(16, 32)
    z, h = torch.randn(4, 8, 16), torch.randn(4, 8, 32)
    plain, _ = owl.layer_outlier_ratios(dense, z, h)
    chunked, _ = owl.layer_outlier_ratios(dense, z, h, batch_chunk=3)
    assert plain == pytest.approx(chunked)


def test_a_layer_with_one_huge_weight_has_a_positive_outlier_ratio():
    dense = _TinyFFN(16, 32)
    with torch.no_grad():
        dense.gate_proj.weight[0, 0] = 1e3
    z, h = torch.randn(2, 8, 16), torch.randn(2, 8, 32)
    ratios, _ = owl.layer_outlier_ratios(dense, z, h, m_grid=(5.0,))
    assert ratios[5.0] > 0.0
