"""オラクルが位置ごとに掛ける重み（``--scored-weight``）。CPU、数秒。

見るのは4点である。

* **重みを渡さない経路が1ビットも動いていないこと。** 既存の測定すべての対照が
  そこにある
* **重みが割合として言った通りであること。** 答え部分の重み s は「目的関数の
  うち答え部分が占める割合」であって、位置1つあたりの倍率ではない
* **印だけを立てた重みが、その位置だけを渡したときと同じ数を出すこと。** 3つの
  オラクルすべてで確かめる。ここが崩れると、重みは「絞った」のではなく「別の
  量を測った」ことになる
* **重みは分割を動かさないこと。** 分割を決めるのは ``--profile-positions`` で、
  この2つは別の軸である
"""

import pytest
import torch

from cmoe.adapters.llama import LlamaAdapter
from cmoe.alloc.oracles.base import LayerWalk, apply_weights
from cmoe.alloc.oracles.local_error import LocalErrorOracle
from cmoe.alloc.oracles.mass import (MassOracle, recovered_mass,
                                     router_selection)
from cmoe.alloc.oracles.suffix_kl import SuffixKLOracle, kl_divergence
from cmoe.assemble import layer_factory
from cmoe.carve.registry import create_carver
from cmoe.data.base import CONTEXT, PAD, SCORED, TokenSet

N_EXPERTS, N_ACTIVE, SEQLEN, BATCH = 4, 3, 16, 2


# -- 重みの作り方（TokenSet） -----------------------------------------------

def marked(scored=(14, 15), pad=(), name='benchqa-fake'):
    """印を持つセット。既定は各系列の末尾2位置が採点位置。"""
    segments = torch.full((BATCH, SEQLEN), CONTEXT, dtype=torch.int8)
    for position in scored:
        segments[:, position] = SCORED
    for position in pad:
        segments[:, position] = PAD
    ids = torch.randint(0, 128, (BATCH, SEQLEN),
                        generator=torch.Generator().manual_seed(0))
    return TokenSet(name, ids, (), (), segments)


def plain():
    """印を持たないセット（素の文章の校正はすべてこれ）。"""
    ids = torch.randint(0, 128, (BATCH, SEQLEN),
                        generator=torch.Generator().manual_seed(1))
    return TokenSet('wikitext2-train-carve', ids)


def test_a_set_without_marks_has_no_weights():
    assert plain().position_weights() is None


def test_a_set_without_marks_refuses_a_weight():
    with pytest.raises(ValueError, match='印を持たない'):
        plain().position_weights(1.0)


def test_the_default_counts_every_real_position_and_no_padding():
    weights = marked(pad=(0, 1)).position_weights()
    assert torch.equal(weights[:, :2], torch.zeros(BATCH, 2))
    assert torch.equal(weights[:, 2:], torch.ones(BATCH, SEQLEN - 2))


@pytest.mark.parametrize('share', [0.0, 0.25, 0.5, 1.0])
def test_the_share_is_what_the_answer_part_occupies(share):
    """答え部分の群が持つ重みの合計が、ちょうど s になっていること。"""
    tokens = marked(pad=(0,))
    weights = tokens.position_weights(share)
    scored = tokens.segments == SCORED
    context = tokens.segments == CONTEXT
    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[scored].sum()) == pytest.approx(share)
    assert float(weights[context].sum()) == pytest.approx(1.0 - share)
    assert float(weights[tokens.segments == PAD].sum()) == 0.0


def test_the_share_that_matches_the_counts_weighs_every_real_position_alike():
    """s = 採点位置の数 / 実位置の数 は、実位置を等しく見るのと同じ重みになる。

    既定（None）との違いは全体の大きさだけで、位置どうしの比は変わらない。
    """
    tokens = marked(pad=(0,))
    n_scored = int((tokens.segments == SCORED).sum())
    n_real = int((tokens.segments != PAD).sum())
    weights = tokens.position_weights(n_scored / n_real)
    flat = weights.reshape(-1)
    positive = flat[flat > 0]
    assert torch.allclose(positive, positive[0].expand_as(positive))
    assert torch.equal(weights > 0, tokens.position_weights() > 0)


@pytest.mark.parametrize('share', [-0.1, 1.5])
def test_a_share_outside_the_unit_interval_is_refused(share):
    with pytest.raises(ValueError, match='0..1'):
        marked().position_weights(share)


def test_an_empty_group_is_refused_when_it_would_carry_weight():
    """答え部分だけを見ると言われたのに採点位置が無いなら、断る。

    黙って文脈だけの数を返すと、走ったはずの実験が別の実験になる。
    """
    segments = torch.full((BATCH, SEQLEN), CONTEXT, dtype=torch.int8)
    tokens = TokenSet('no-scored', torch.zeros(BATCH, SEQLEN, dtype=torch.long),
                      (), (), segments)
    with pytest.raises(ValueError, match='採点位置が1つも無い'):
        tokens.position_weights(1.0)
    with pytest.raises(ValueError, match='文脈位置が1つも無い'):
        marked(scored=tuple(range(SEQLEN))).position_weights(0.0)


# -- 重みの掛け方 -----------------------------------------------------------

def test_no_weights_returns_the_very_same_tensor():
    per_token = torch.arange(5, dtype=torch.float32)
    assert apply_weights(per_token, None) is per_token


def test_weights_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match='対応していない'):
        apply_weights(torch.ones(5), torch.ones(4))


# -- KL の重み付け -----------------------------------------------------------

@pytest.fixture
def logits():
    generator = torch.Generator().manual_seed(2)
    reference = torch.randn(BATCH, SEQLEN, 32, generator=generator)
    candidate = reference + 0.1 * torch.randn(BATCH, SEQLEN, 32,
                                              generator=generator)
    return reference, candidate


def test_equal_weights_are_the_plain_mean(logits):
    reference, candidate = logits
    plain_kl, _ = kl_divergence(reference, candidate)
    weighted, _ = kl_divergence(reference, candidate,
                                weights=torch.ones(BATCH * SEQLEN))
    assert weighted == pytest.approx(plain_kl, rel=1e-9)


def test_marking_a_subset_is_the_kl_of_that_subset(logits):
    """印だけを立てた重みは、その位置だけを渡して測ったのと同じ数を出す。"""
    reference, candidate = logits
    rows = torch.zeros(BATCH * SEQLEN, dtype=torch.bool)
    rows[3:9] = True
    weights = rows.to(torch.float32)
    flat_reference = reference.reshape(-1, 32)
    flat_candidate = candidate.reshape(-1, 32)
    subset, n_subset = kl_divergence(flat_reference[rows], flat_candidate[rows])
    weighted, n_weighted = kl_divergence(reference, candidate, weights=weights)
    assert weighted == pytest.approx(subset, rel=1e-9)
    # 数えた位置の数も、印の数そのものを報告する
    assert n_weighted == n_subset == int(rows.sum())


def test_the_weight_scale_does_not_matter(logits):
    """重みは相対値である（合計で割るので、倍にしても値は変わらない）。"""
    reference, candidate = logits
    weights = torch.rand(BATCH * SEQLEN, generator=torch.Generator().manual_seed(3))
    one, _ = kl_divergence(reference, candidate, weights=weights)
    twice, _ = kl_divergence(reference, candidate, weights=weights * 2)
    assert twice == pytest.approx(one, rel=1e-9)


def test_weights_that_do_not_match_the_readout_are_refused(logits):
    reference, candidate = logits
    with pytest.raises(ValueError, match='対応していない'):
        kl_divergence(reference, candidate, weights=torch.ones(3))


def test_weights_that_are_all_zero_are_refused(logits):
    reference, candidate = logits
    with pytest.raises(ValueError, match='採点する位置が無い'):
        kl_divergence(reference, candidate,
                      weights=torch.zeros(BATCH * SEQLEN))


# -- オラクルまで通す --------------------------------------------------------

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


def make_walk(adapter, tokens, weights=None):
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    factory = layer_factory(create_carver('cmoe', N_EXPERTS), N_EXPERTS,
                            device=adapter.device)
    return LayerWalk(adapter, inputs, factory, N_EXPERTS,
                     n_active_total=N_ACTIVE, score_weights=weights)


def measure_one(oracle, x=2, layer=0):
    walk = oracle.walk
    state = walk.root()
    profile = walk.profile(state, layer)
    carved = walk.carve(profile, x)
    child = walk.propagate(profile, carved)
    return oracle.measure(profile, carved, state, child), profile, carved, child


def subset_weights(positions):
    weights = torch.zeros(BATCH, SEQLEN)
    weights[:, positions] = 1.0
    return weights


def test_the_walk_refuses_weights_of_the_wrong_shape(adapter):
    with pytest.raises(ValueError, match='対応していない'):
        make_walk(adapter, plain(), torch.ones(BATCH, SEQLEN + 1))


def test_the_walk_refuses_weights_that_count_nothing(adapter):
    with pytest.raises(ValueError, match='採点する位置が無い'):
        make_walk(adapter, plain(), torch.zeros(BATCH, SEQLEN))


@pytest.mark.parametrize('oracle_class', [SuffixKLOracle, MassOracle,
                                          LocalErrorOracle])
def test_equal_weights_score_the_same_as_no_weights(adapter, oracle_class):
    """全位置に同じ重みを置くのは、重みを渡さないのと同じ採点である。"""
    tokens = plain()
    bare = oracle_class(make_walk(adapter, tokens))
    even = oracle_class(make_walk(adapter, tokens, torch.ones(BATCH, SEQLEN)))
    assert (measure_one(even)[0].score
            == pytest.approx(measure_one(bare)[0].score, rel=1e-6))


def test_the_suffix_kl_scores_only_the_marked_positions(adapter):
    """印を立てた suffix KL は、その位置だけで取った KL と同じ数になる。

    重みが 0 の位置の読み出しは持ちさえしないので、これが崩れているなら
    「捨てた位置」と「数える位置」がずれている。
    """
    tokens = plain()
    positions = [2, 5, 11]
    weights = subset_weights(positions)

    bare = SuffixKLOracle(make_walk(adapter, tokens))
    _, profile, _, child = measure_one(bare)
    full = adapter.forward_suffix(profile.layer + 1, child.hidden,
                                  bare.walk.inputs)
    vocab = full.shape[-1]
    rows = weights.reshape(-1) > 0
    expected, _ = kl_divergence(bare.dense_logits.reshape(-1, vocab)[rows],
                                full.reshape(-1, vocab)[rows])

    weighted = SuffixKLOracle(make_walk(adapter, tokens, weights))
    assert measure_one(weighted)[0].score == pytest.approx(expected, rel=1e-6)


def test_the_suffix_kl_keeps_only_the_readout_it_counts(adapter):
    """数えない位置の読み出しは並べない（確保の上限を決めているのはここ）。"""
    weights = subset_weights([2, 5, 11])
    oracle = SuffixKLOracle(make_walk(adapter, plain(), weights))
    assert oracle.dense_logits.shape[0] == int(weights.sum())
    assert oracle.dense_logits.dim() == 2


@pytest.mark.parametrize('squared', [False, True])
def test_the_mass_oracle_counts_the_marked_positions_only(adapter, squared):
    tokens = plain()
    positions = [1, 7, 13]
    weights = subset_weights(positions)
    walk = make_walk(adapter, tokens, weights)
    oracle = MassOracle(walk, squared=squared)
    result, profile, carved, _ = measure_one(oracle)

    h = walk.true_activations(profile)
    indices = router_selection(carved.moe, profile.z)
    rows = weights.reshape(-1) > 0
    expected, _ = recovered_mass(h[rows], carved.partition, indices[rows],
                                 squared)
    assert result.details['r'] == pytest.approx(expected, rel=1e-6)
    assert result.details['n_tokens'] == int(weights.sum())


def test_the_local_error_counts_the_marked_positions_only(adapter):
    tokens = plain()
    weights = subset_weights([0, 4, 9])
    rows = weights.reshape(-1) > 0

    weighted = LocalErrorOracle(make_walk(adapter, tokens, weights))
    result, profile, carved, _ = measure_one(weighted)

    # 同じ層・同じ候補を、その位置だけの H で測り直す
    from cmoe.alloc.oracles.local_error import (exact_fp32_matmul,
                                                missed_energy, output_energy)
    from cmoe.alloc.oracles.mass import neuron_to_expert

    h = weighted.walk.true_activations(profile)[rows]
    indices = router_selection(carved.moe, profile.z)[rows]
    w32 = profile.dense.down_proj.weight.to(torch.float32)
    group_of = neuron_to_expert(carved.partition, h.shape[1], w32.device)
    with exact_fp32_matmul():
        total = output_energy(h, w32)
        missed = missed_energy(h, w32, group_of, indices,
                               len(carved.partition.routed_groups))
    expected = float(missed.sum() / total.sum())
    assert result.details['l'] == pytest.approx(expected, rel=1e-6)
    assert result.details['n_tokens'] == int(weights.sum())


def test_the_weights_do_not_move_the_split(adapter):
    """重みは採点だけを動かす。分割は ``--profile-positions`` の仕事である。

    ここが動くなら、2つの軸が混ざっていて、勝ったときにどちらが効いたのか
    言えなくなる。
    """
    tokens = plain()
    bare = make_walk(adapter, tokens)
    weighted = make_walk(adapter, tokens, subset_weights([3, 6]))
    left = bare.carve(bare.profile(bare.root(), 0), 2)
    right = weighted.carve(weighted.profile(weighted.root(), 0), 2)
    assert left.partition.shared_group == right.partition.shared_group
    assert left.partition.routed_groups == right.partition.routed_groups


# -- 読み出しを絞る（アダプタ） ----------------------------------------------

def test_keeping_positions_returns_the_same_rows_as_selecting_afterwards(adapter):
    """``forward_suffix(keep=...)`` は、全部作ってから選ぶのと同じ行を返す。"""
    tokens = plain()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    keep = torch.zeros(BATCH, SEQLEN, dtype=torch.bool)
    keep[0, 3] = keep[1, 8] = keep[1, 15] = True

    whole = adapter.forward_suffix(0, inputs.hidden, inputs)
    kept = adapter.forward_suffix(0, inputs.hidden, inputs, keep=keep)
    assert torch.equal(kept, whole.reshape(-1, whole.shape[-1])[keep.reshape(-1)])


def test_keeping_positions_survives_being_split_into_batches(adapter):
    tokens = plain()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    keep = torch.zeros(BATCH, SEQLEN, dtype=torch.bool)
    keep[0, 3] = keep[1, 8] = True
    whole = adapter.forward_suffix(0, inputs.hidden, inputs, keep=keep)
    split = adapter.forward_suffix(0, inputs.hidden, inputs, batch_chunk=1,
                                   keep=keep)
    assert torch.equal(whole, split)


def test_a_keep_mask_of_the_wrong_shape_is_refused(adapter):
    tokens = plain()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    with pytest.raises(ValueError, match='対応していない'):
        adapter.forward_suffix(0, inputs.hidden, inputs,
                               keep=torch.ones(BATCH, SEQLEN + 1,
                                               dtype=torch.bool))


# -- CLI の断り方 ------------------------------------------------------------

def parse(*extra):
    from cmoe.cli import build_parser

    return build_parser().parse_args(['search', *extra])


def test_the_cli_refuses_a_share_outside_the_unit_interval():
    from cmoe.cli import check_oracle_arguments

    with pytest.raises(SystemExit, match=r'0\.\.1'):
        check_oracle_arguments(parse('--scored-weight', '1.5'))


def test_the_cli_refuses_a_calibration_set_without_marks():
    """印を持たないセットに重みを指定したら、走らせる前に断る。"""
    from cmoe.cli import score_weights

    with pytest.raises(SystemExit, match='印を持たない'):
        score_weights(parse('--scored-weight', '1.0'), plain())


def test_the_cli_asks_for_no_weights_on_a_plain_set():
    from cmoe.cli import score_weights

    assert score_weights(parse(), plain()) is None


def test_the_cli_drops_the_padding_even_without_a_share():
    """印を持つセットでは、重みを指定しなくても埋めは数えない。"""
    from cmoe.cli import score_weights

    tokens = marked(pad=(0, 1))
    weights = score_weights(parse(), tokens)
    assert torch.equal(weights > 0, tokens.segments != PAD)
