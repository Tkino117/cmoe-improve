"""Llama 以外のモデルでも同じ経路が通ることの確認（CPU、極小）。

モデル差異がアダプタに閉じているかを確かめるためのテスト。組み立て役より上は
1行も変えずに Qwen2 を変換して評価できることを見る。
"""

import torch
import pytest

from cmoe.adapters.auto import AutoAdapter
from cmoe.alloc.base import Allocation
from cmoe.assemble import Converter
from cmoe.carve.registry import create_carver
from cmoe.data.base import TokenSet
from cmoe.eval.ppl import evaluate_ppl
from cmoe.router.registry import create_method

N_EXPERTS, N_ACTIVE, SEQLEN = 4, 4, 16


@pytest.fixture
def qwen_adapter():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN * 8)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    model.eval()
    model.config.use_cache = False
    return AutoAdapter(model, seqlen=SEQLEN, device='cpu').check_shape()


def token_set(name, shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TokenSet(name, torch.randint(0, 128, shape, generator=generator))


def test_qwen2_converts_through_the_same_path(qwen_adapter):
    converter = Converter(
        qwen_adapter, create_carver('cmoe', N_EXPERTS), [create_method('cmoe')],
        n_experts=N_EXPERTS)
    allocation = Allocation((1, 3), name='mixed', n_active_total=N_ACTIVE)
    report = converter.convert(token_set('calib', (2, SEQLEN)), allocation)

    assert [row.topk for row in report.layers] == [3, 1]
    assert all(qwen_adapter.is_converted(index) for index in range(2))
    assert evaluate_ppl(qwen_adapter, token_set('eval', (1, SEQLEN * 2), seed=1)).ppl > 0


def test_shape_check_rejects_a_model_that_does_not_fit(qwen_adapter):
    del qwen_adapter.layers[0].mlp.gate_proj
    with pytest.raises(ValueError, match='gate_proj'):
        qwen_adapter.check_shape()
