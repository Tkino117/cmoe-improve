"""ルーター方式の CPU スモーク（極小モデル）。

数値の一致は GPU のアンカー（experiments/00_anchor.py）で見る。ここで見るのは
方式が契約を守っているか — 分割を変えない、代表が自分の expert の中にいる、
方式4 が出発点より悪くならない、診断が回収率の上限関係を満たす — である。
"""

import math
from types import SimpleNamespace

import torch
import pytest

from cmoe.adapters.llama import LlamaAdapter
from cmoe.alloc.base import Allocation
from cmoe.assemble import Converter, install_routers
from cmoe.carve.registry import create_carver
from cmoe.data.base import TokenSet
from cmoe.eval.ppl import evaluate_ppl
from cmoe.router.registry import create_method, resolve_chain

# A < N なので Top-K が routed 数より小さくなる（方式4 の探索が動く条件）
N_EXPERTS, N_ACTIVE, X, SEQLEN = 4, 3, 1, 16


@pytest.fixture
def adapter():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN * 8)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.eval()
    model.config.use_cache = False
    return LlamaAdapter(model, seqlen=SEQLEN, device='cpu')


def token_set(name, shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TokenSet(name, torch.randint(0, 128, shape, generator=generator))


def convert(adapter, names, diagnostics=True):
    methods = [create_method(name) for name in resolve_chain(names)]
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), methods,
        n_experts=N_EXPERTS, fit_batch_chunk=2, token_chunk=32)
    allocation = Allocation(tuple([X] * adapter.n_layers),
                            name=f'uniform{X}', n_active_total=N_ACTIVE)
    return converter.convert(
        token_set('calib', (2, SEQLEN)),
        allocation,
        fit=token_set('fit', (2, SEQLEN), seed=2),
        validation=token_set('val', (2, SEQLEN), seed=3) if diagnostics else None,
    )


def test_recovery_chain_builds_every_method(adapter):
    report = convert(adapter, ['oracle_recovery'])

    assert report.router_methods == (
        'cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery')
    for name in report.router_methods:
        assert len(report.routers[name]) == adapter.n_layers

    for record in report.layers:
        routed = record.expert_sizes[1:]
        assert record.topk == N_ACTIVE - X
        # どの方式の代表も、自分の expert の大きさの範囲に収まる本数である
        for name, row in record.representatives.items():
            assert row is not None and len(row) == len(routed), name


def test_recovery_is_never_worse_than_its_starting_points(adapter):
    report = convert(adapter, ['oracle_recovery'])

    for record in report.layers:
        selection = record.selections['oracle_recovery']
        assert selection['fit_recovery'] >= max(selection['initial_recoveries'])
        assert selection['starts'], '出発点が1つも無い'
        for start in selection['starts']:
            assert start['final_recovery'] >= start['initial_recovery']


def test_diagnostics_bracket_the_router_by_the_oracle(adapter):
    report = convert(adapter, ['oracle_recovery'])

    for record in report.layers:
        for name, values in record.diagnostics.items():
            # オラクルは同じ分割の上での上限なので、下回ることはない
            assert values['router_r'] <= values['oracle_r'] + 1e-9, name
            assert 0.0 <= values['oracle_mean_recall'] <= 1.0
            assert 0.0 <= values['oracle_exact_set_rate'] <= 1.0


def test_expert_mean_reports_no_representative(adapter):
    report = convert(adapter, ['expert_mean'], diagnostics=False)

    assert report.router_methods == ('cmoe', 'expert_mean')
    for record in report.layers:
        assert record.representatives['expert_mean'] is None
        assert record.representatives['cmoe'] is not None


def test_installing_a_router_changes_only_the_gate(adapter):
    report = convert(adapter, ['oracle_recovery'], diagnostics=False)
    experts_before = [adapter.layers[i].mlp.experts for i in range(adapter.n_layers)]

    baseline = evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 2), seed=1)).ppl
    install_routers(adapter, report.routers['oracle_recovery'])
    after = evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 2), seed=1)).ppl

    assert baseline > 0 and after > 0
    for index, experts in enumerate(experts_before):
        assert adapter.layers[index].mlp.experts is experts
        assert adapter.layers[index].mlp.gate is report.routers['oracle_recovery'][index]


def test_topk_zero_layers_share_the_baseline_router(adapter):
    """x が A と等しい層では Top-K=0 になり、どの方式も出力を変えられない。"""
    methods = [create_method(name) for name in resolve_chain(['oracle_recovery'])]
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), methods, n_experts=N_EXPERTS,
        fit_batch_chunk=2, token_chunk=32)
    allocation = Allocation((N_ACTIVE, X), name='mixed', n_active_total=N_ACTIVE)
    report = converter.convert(
        token_set('calib', (2, SEQLEN)), allocation,
        fit=token_set('fit', (2, SEQLEN), seed=2))

    assert report.layers[0].topk == 0
    assert report.layers[0].diagnostics == {}
    assert (report.routers['oracle_recovery'][0] is report.routers['cmoe'][0])
    # routing が残っている層では別のモジュールになる
    assert (report.routers['oracle_recovery'][1] is not report.routers['cmoe'][1])


def test_fit_is_required_by_the_methods_that_declare_it(adapter):
    methods = [create_method(name) for name in resolve_chain(['oracle_correlation'])]
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), methods, n_experts=N_EXPERTS)
    allocation = Allocation(tuple([X] * adapter.n_layers), n_active_total=N_ACTIVE)
    with pytest.raises(ValueError, match='fit'):
        converter.convert(token_set('calib', (2, SEQLEN)), allocation)


def test_resolve_chain_orders_the_dependencies():
    assert resolve_chain(['oracle_recovery']) == [
        'cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery']
    assert resolve_chain(['expert_mean']) == ['cmoe', 'expert_mean']
    assert resolve_chain(['cmoe']) == ['cmoe']
    # 明示的に混ぜても、方式4 の依存は前に来る
    assert resolve_chain(['expert_mean', 'oracle_recovery'])[:4] == [
        'cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery']


def test_score_calibration_freezes_the_recovery_representatives(adapter):
    """方式5 は代表を1つも動かさない。動かすのは gain と offset だけ。"""
    report = convert(adapter, ['score_calibration'], diagnostics=False)

    assert report.router_methods == (
        'cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery',
        'score_calibration')
    for record in report.layers:
        assert (record.representatives['score_calibration']
                == record.representatives['oracle_recovery'])


def test_score_calibration_never_loses_to_its_frozen_start(adapter):
    report = convert(adapter, ['score_calibration'], diagnostics=False)

    for record in report.layers:
        selection = record.selections['score_calibration']
        # 受理は直接測った改善だけなので、単調性は仮定ではなく構成で出る
        assert selection['fit_recovery'] >= selection['initial_recovery']
        assert selection['gain_recovery'] >= selection['initial_recovery']
        assert selection['fit_recovery'] >= selection['gain_recovery']
        for move in selection['accepted_moves']:
            assert move['stage'] in ('gain', 'bias')
        low, high = 1.0 / selection['gain_limit'], selection['gain_limit']
        for gain in selection['gains']:
            assert low <= gain <= high


def test_score_calibration_router_reproduces_the_fitted_selection(adapter):
    """載せたルーターが、fit が採点したつまみをそのまま持っている。"""
    from cmoe.router.methods.score_calibration import scaled_up_rows

    report = convert(adapter, ['score_calibration'], diagnostics=False)
    for index, router in enumerate(report.routers['score_calibration']):
        selection = report.layers[index].selections.get('score_calibration')
        if selection is None:      # Top-K=0 の層は基準ルーターを共有する
            continue
        frozen = report.routers['oracle_recovery'][index]
        assert router.representative_indices == tuple(selection['representatives'])
        assert torch.equal(
            router.extra_bias.cpu(),
            torch.tensor(selection['biases'], dtype=router.extra_bias.dtype))
        # gate 行は方式4 のまま。動くのは classifier 行の大きさだけ
        assert torch.equal(router.gate.weight.data, frozen.gate.weight.data)
        assert torch.equal(
            router.classifier.weight.data,
            scaled_up_rows(frozen.classifier.weight.data, selection['gains']))


def test_best_cut_is_the_exact_argmax_of_the_step_function():
    """階段関数の厳密な argmax。格子ではなく切り口を数え上げて確かめる。"""
    from cmoe.router.methods.score_calibration import best_cut

    thresholds = torch.tensor([0.5, 2.0, 1.0, 3.0])
    delta = torch.tensor([1.0, -5.0, 2.0, 1.0])
    interval = best_cut(thresholds, delta, (-math.inf, math.inf))

    # τ<knob のトークンを足すので、0.5 と 1.0 を取り 2.0 を避ける区間が勝つ
    assert interval == (1.0, 2.0)
    knob = sum(interval) / 2
    best = float(delta[thresholds < knob].sum())
    for other in (0.0, 0.75, 2.5, 3.5):
        assert float(delta[thresholds < other].sum()) <= best


def test_best_cut_stays_inside_the_bounds_it_is_given():
    from cmoe.router.methods.score_calibration import best_cut

    thresholds = torch.tensor([0.5, 2.0, 1.0, 3.0])
    delta = torch.tensor([1.0, 5.0, 2.0, 1.0])
    # 制約なしなら 3.0 より上が勝つが、上限がそこへ届かせない
    assert best_cut(thresholds, delta, (-math.inf, math.inf)) == (3.0, math.inf)
    low, high = best_cut(thresholds, delta, (0.0, 1.5))
    assert 0.0 <= low < high <= 1.5


def test_score_calibration_needs_exactly_one_frozen_set(adapter):
    """凍結する相手が構築順の前に居ないと作れない。"""
    from cmoe.router.methods.score_calibration import ScoreCalibrationMethod

    method = ScoreCalibrationMethod()
    context = object()
    with pytest.raises(ValueError, match='代表集合1本'):
        method.build(SimpleNamespace(initial_representative_sets=()), context)


def test_resolve_chain_puts_the_frozen_source_first():
    assert resolve_chain(['score_calibration']) == [
        'cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery',
        'score_calibration']
