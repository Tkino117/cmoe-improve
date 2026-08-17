"""ルーター方式の CPU スモーク（極小モデル）。

数値の一致は GPU のアンカー（experiments/00_anchor.py）で見る。ここで見るのは
方式が契約を守っているか — 分割を変えない、代表が自分の expert の中にいる、
方式4 が出発点より悪くならない、診断が回収率の上限関係を満たす — である。
"""

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
