"""静的プルーニングの対照の CPU スモーク。

見るのは3つだけ。刈った形が意図どおりか、パラメータ会計が手計算と合うか、
刈ったモデルが評価経路を最後まで通るか。**手法の質はここでは見ない**
（極小モデルの WIFV や Taylor 重要度に意味は無い）。
"""

import torch
import pytest

from cmoe.adapters.llama import LlamaAdapter
from cmoe.data.base import TokenSet
from cmoe.eval.ppl import evaluate_ppl
from cmoe.prune import flap, llm_pruner
from cmoe.prune.base import (accounting, apply_plan, block_params, model_shape,
                             moe_activated_sparsity)

SEQLEN = 16


@pytest.fixture
def adapter():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN * 8)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.float32)
    model.eval()
    model.config.use_cache = False
    return LlamaAdapter(model, seqlen=SEQLEN, device='cpu')


def token_set(name, shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TokenSet(name, torch.randint(0, 128, shape, generator=generator))


def test_moe_reference_sparsity():
    """N=8 / A=6 の「25%」がブロック全体では 16.71% であること。

    これが対照のスパース率を読み替える根拠なので、数字そのものを固定する。
    """
    shape = {'hidden_size': 4096, 'intermediate_size': 11008}
    parts = block_params(shape)
    assert parts['ffn'] == 3 * 4096 * 11008
    assert parts['attn'] == 4 * 4096 * 4096
    value = moe_activated_sparsity(shape, 8, 6)
    assert value == pytest.approx(0.1671, abs=1e-4)
    assert moe_activated_sparsity(shape, 8, 4) == pytest.approx(0.3342, abs=1e-4)


def test_fluctuation_matches_the_published_arithmetic():
    """``_BiasStats`` が FLAP 公式の ``BiasGPT`` と同じ数字を出すこと。

    公式実装の ``old_baseline_inp`` は**別名であって複製ではなく**、直後の
    in-place 更新で新しい平均に変わる。したがって掛け合わせる2因子は同じもので、
    実体は「更新後の平均からの偏差の2乗和」である。clone を挟むと別の統計量に
    なるので、ここを固定しておく（公式実装との突き合わせでも差 0 を確認済み）。
    """
    torch.manual_seed(0)
    stats = flap._BiasStats(4, torch.device('cpu'))
    baseline = torch.zeros(4)
    fluc = torch.zeros(4)
    for n in range(5):
        sample = torch.randn(1, 3, 4)
        values = sample.reshape(-1, 4).t()
        baseline = baseline * (n / (n + 1)) + values.mean(dim=1) / (n + 1)
        if n == 0:
            fluc = torch.zeros(4)
        else:
            deviation = values - baseline.unsqueeze(1)
            fluc = fluc * ((n - 1) / n) + (deviation * deviation).sum(dim=1) / (n + 1)
        stats.add(sample)
    assert torch.allclose(stats.baseline_inp, baseline)
    assert torch.allclose(stats.fluc_inp, fluc)


@pytest.mark.parametrize('scope', ['block', 'mlp'])
def test_flap_plan_and_evaluate(adapter, scope):
    calibration = token_set('calib', (4, SEQLEN))
    plan = flap.build_plan(adapter, calibration, 0.25, scope=scope)
    shape = model_shape(adapter)
    counts = accounting(plan, shape)

    if scope == 'mlp':
        # FFN だけを刈るので、FFN ニューロンの割合がそのまま目標に一致する
        assert counts['ffn_neuron_sparsity'] == pytest.approx(0.25, abs=0.01)
        assert counts['removed_attn_params'] == 0
        assert counts['mean_heads'] == shape['n_heads']
    else:
        # attn+FFN 全体の比が目標に一致する（AL-AM の大域しきい値）
        assert counts['effective_block_sparsity'] == pytest.approx(0.25, abs=0.02)
    # 層ごとに残す本数が変わる（AL = adaptive layer）
    assert len(set(counts['intermediate_per_layer'])) > 1

    apply_plan(adapter, plan)
    for index, kept in enumerate(counts['intermediate_per_layer']):
        mlp = adapter.layers[index].mlp
        assert mlp.gate_proj.weight.shape[0] == kept
        assert mlp.down_proj.weight.shape[1] == kept
        # FLAP は落としたチャネルの平均寄与を bias に移す
        assert mlp.down_proj.bias is not None
    assert evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 4))).ppl > 0


def test_llm_pruner_plan_and_evaluate(adapter):
    calibration = token_set('calib', (2, SEQLEN))
    plan = llm_pruner.build_plan(adapter, calibration, 0.25, scope='block',
                                 layer_start=1, layer_end=3)
    shape = model_shape(adapter)
    counts = accounting(plan, shape)

    # 刈るのは層1・2 だけ。層0・3 は手つかず
    assert counts['intermediate_per_layer'] == [128, 96, 96, 128]
    assert counts['heads_per_layer'] == [4, 3, 3, 4]
    # 名目 25% でも、4層中2層しか刈らないので実効は半分になる
    assert counts['effective_block_sparsity'] == pytest.approx(0.125, abs=0.01)
    assert plan.knobs['recovery_tuning'] is False

    # 勾配を残したままにしない（7B では 13.5GB がそのまま残る）
    assert all(param.grad is None for param in adapter.model.parameters())

    apply_plan(adapter, plan)
    assert adapter.layers[1].self_attn.num_heads == 3
    assert adapter.layers[0].self_attn.num_heads == 4
    assert evaluate_ppl(adapter, token_set('eval', (1, SEQLEN * 4))).ppl > 0


def test_gradient_chunking_gives_the_same_plan(adapter):
    """勾配を塊に割っても、1回でやったのと同じ計画になること。

    校正を長い系列に揃えると 32k トークンを1回の backward に載せることになり、
    活性が載らない。割っても値が変わらないことが、割ってよい根拠である。
    """
    calibration = token_set('calib', (4, SEQLEN))
    whole = llm_pruner.build_plan(adapter, calibration, 0.25, layer_start=1,
                                  layer_end=3, token_chunk=SEQLEN * 4)
    split = llm_pruner.build_plan(adapter, calibration, 0.25, layer_start=1,
                                  layer_end=3, token_chunk=SEQLEN)
    assert whole.knobs['backward_loss'] == pytest.approx(
        split.knobs['backward_loss'], rel=1e-5)
    assert [row.mlp_keep for row in whole.layers] == \
        [row.mlp_keep for row in split.layers]
    assert [row.head_keep for row in whole.layers] == \
        [row.head_keep for row in split.layers]


def test_llm_pruner_mlp_scope_raises_the_per_layer_ratio(adapter):
    """FFN 全体の 25% を半分の層だけで落とすなら、層あたりは 50% になる。"""
    calibration = token_set('calib', (2, SEQLEN))
    plan = llm_pruner.build_plan(adapter, calibration, 0.25, scope='mlp',
                                 layer_start=1, layer_end=3)
    counts = accounting(plan, model_shape(adapter))
    assert plan.knobs['channel_ratio'] == pytest.approx(0.5)
    assert counts['ffn_neuron_sparsity'] == pytest.approx(0.25, abs=0.01)
    assert counts['removed_attn_params'] == 0


def test_refuses_gqa_models():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=SEQLEN * 8)
    model = LlamaForCausalLM(config).to(torch.float32)
    model.config.use_cache = False
    adapter = LlamaAdapter(model, seqlen=SEQLEN, device='cpu')
    with pytest.raises(ValueError, match='GQA'):
        llm_pruner.build_plan(adapter, token_set('calib', (2, SEQLEN)), 0.25)
