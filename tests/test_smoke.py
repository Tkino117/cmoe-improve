"""極小モデルの CPU スモーク。

数値は見ない。組み立て役から評価までが最後まで走ること、つまり API を壊して
いないことだけを数秒で確かめる。GPU で本番を回す前段。
"""

import torch
import pytest

from cmoe.adapters.llama import LlamaAdapter
from cmoe.alloc.base import Allocation
from cmoe.assemble import Converter
from cmoe.carve.registry import create_carver
from cmoe.data.base import TokenSet
from cmoe.eval.ppl import evaluate_ppl
from cmoe.router.registry import create_method

N_EXPERTS, N_ACTIVE, SEQLEN = 4, 4, 16


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


def convert(adapter, x, router='cmoe'):
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), [create_method(router)],
        n_experts=N_EXPERTS)
    allocation = Allocation(tuple([x] * adapter.n_layers), name=f'uniform{x}',
                            n_active_total=N_ACTIVE)
    return converter.convert(token_set('calib', (2, SEQLEN)), allocation)


@pytest.mark.parametrize('x', [0, 1, 3])
def test_convert_and_evaluate(adapter, x):
    report = convert(adapter, x)

    assert len(report.layers) == adapter.n_layers
    for row in report.layers:
        assert row.n_shared == x
        assert row.topk == N_ACTIVE - x
        # 分割は FFN のニューロンを覆い尽くす（余りが消えていない）
        assert sum(row.expert_sizes) == 128
    assert all(adapter.is_converted(index) for index in range(adapter.n_layers))

    result = evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 4), seed=1))
    assert result.n_chunks == 4
    assert len(result.chunk_mean_nlls) == 4
    assert result.ppl > 0 and torch.isfinite(torch.tensor(result.ppl))


def test_allocation_keeps_total_active_fixed():
    allocation = Allocation((0, 2, 4), n_active_total=4)
    assert [allocation.topk(i) for i in range(3)] == [4, 2, 0]
    with pytest.raises(ValueError):
        Allocation((5,), n_active_total=4)


def test_per_layer_allocation_reaches_the_layers(adapter):
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), [create_method('cmoe')],
        n_experts=N_EXPERTS)
    allocation = Allocation((1, 3), name='mixed', n_active_total=N_ACTIVE)
    report = converter.convert(token_set('calib', (2, SEQLEN)), allocation)

    assert [row.n_shared for row in report.layers] == [1, 3]
    assert [row.topk for row in report.layers] == [3, 1]
    assert [adapter.layers[i].mlp.gate.topk for i in range(2)] == [3, 1]


def test_router_method_may_not_change_the_partition(adapter):
    class Broken:
        name = 'broken'
        training_free = True

        def build(self, context, baseline):
            from cmoe.moe.modules import Router
            return Router(context.hidden_size, context.n_routed, context.topk + 1)

    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), [Broken()], n_experts=N_EXPERTS)
    allocation = Allocation(tuple([1] * adapter.n_layers), n_active_total=N_ACTIVE)
    with pytest.raises(ValueError, match='topk'):
        converter.convert(token_set('calib', (2, SEQLEN)), allocation)


def test_partial_conversion_leaves_the_rest_dense(adapter):
    """--layers 用の経路。先頭だけ変換し、残りは dense のまま動く。"""
    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), [create_method('cmoe')],
        n_experts=N_EXPERTS, n_layers=1)
    allocation = Allocation((2,), name='uniform2', n_active_total=N_ACTIVE)
    report = converter.convert(token_set('calib', (2, SEQLEN)), allocation)

    assert len(report.layers) == 1
    assert adapter.is_converted(0)
    assert not adapter.is_converted(1)
    assert evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 2), seed=1)).ppl > 0


def test_a_batch_chunk_wider_than_the_batch_still_moves_the_states(adapter):
    """分割幅が系列数以上でも、載せ替えを省かない。

    分割を頼まれているとき、呼ぶ側は状態をホストに置いている。1塊で済むから
    といってそのまま層へ渡すと、系列数が分割幅を下回る校正セットでだけ
    「重みはカード、状態はホスト」で落ちる。CPU では落ちないので、ここで見るのは
    「分割幅を変えても同じ値が出る」ことである。
    """
    from cmoe.moe.modules import forward_chunked

    # 中身は何でもよい。見ているのは分割の枝であって、層の実体ではない
    hidden = adapter.model.config.hidden_size
    torch.manual_seed(0)
    moe = torch.nn.Linear(hidden, hidden, bias=False).to(torch.bfloat16)
    z = torch.randn(3, SEQLEN, hidden, dtype=torch.bfloat16)
    residual = torch.zeros_like(z)
    wide = forward_chunked(moe, z, residual, batch_chunk=8, device='cpu')
    narrow = forward_chunked(moe, z, residual, batch_chunk=2, device='cpu')
    assert torch.equal(wide, narrow)
    assert wide.device == z.device
