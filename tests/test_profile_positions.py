"""活性を数える位置を絞る（``carve.profile.select_positions``）。CPU、数秒。

見るのは3点である。**絞らない経路が1ビットも動いていないこと**（既存の測定
すべての対照がそれである）、**絞った統計が、その位置だけを渡したときと同じ
であること**、そして**組み立て役が絞るのはプロファイルだけで、次の層へ渡す
出力は絞っていない z から作ること**である。
"""

import pytest
import torch

from cmoe.carve.profile import profile_layer, select_positions

HIDDEN, INTER, BATCH, SEQ = 8, 32, 3, 7


@pytest.fixture
def layer():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP

    config = LlamaConfig(hidden_size=HIDDEN, intermediate_size=INTER)
    torch.manual_seed(0)
    dense = LlamaMLP(config).eval()
    z = torch.randn(BATCH, SEQ, HIDDEN,
                    generator=torch.Generator().manual_seed(1))
    return dense, z


@pytest.fixture
def mask():
    """位置の半分ほどに印を付ける。系列ごとに違う場所にする。"""
    grid = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    grid[0, 1:4] = True
    grid[1, 0] = True
    grid[2, 2:] = True
    return grid


# --- 絞らない経路 -----------------------------------------------------------

def test_no_mask_returns_the_very_same_tensor(layer):
    _, z = layer
    assert select_positions(z, None) is z


def test_no_mask_leaves_the_statistics_untouched(layer):
    dense, z = layer
    rates, markers = profile_layer(dense, z, k_act=4)
    same_rates, same_markers = profile_layer(
        dense, select_positions(z, None), k_act=4)
    assert torch.equal(rates, same_rates)
    assert torch.equal(markers, same_markers)


# --- 絞った経路 -------------------------------------------------------------

def test_the_marked_positions_are_taken_in_row_major_order(layer, mask):
    _, z = layer
    picked = select_positions(z, mask)
    assert picked.shape == (1, int(mask.sum()), HIDDEN)
    assert torch.equal(picked[0], z.reshape(-1, HIDDEN)[mask.reshape(-1)])


def test_the_statistics_match_handing_over_only_those_positions(layer, mask):
    """絞ったあとの統計は、その位置だけの列を渡したときと同じでなければならない。

    プロファイルは ``[トークン, ニューロン]`` に潰してから数えるので、系列の
    形が変わっても値は動かない。ここが崩れると、絞ったことと形を変えたことが
    混ざる。
    """
    dense, z = layer
    rows = z.reshape(-1, HIDDEN)[mask.reshape(-1)]
    expected = profile_layer(dense, rows.reshape(-1, 1, HIDDEN), k_act=4)
    actual = profile_layer(dense, select_positions(z, mask), k_act=4)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])


def test_the_rate_is_the_share_among_the_marked_positions_only(layer, mask):
    dense, z = layer
    rates, markers = profile_layer(dense, select_positions(z, mask), k_act=4)
    assert markers.shape == (int(mask.sum()), INTER)
    assert torch.allclose(rates, markers.mean(dim=0))


def test_marking_everything_is_the_same_as_not_marking(layer):
    dense, z = layer
    every = torch.ones(BATCH, SEQ, dtype=torch.bool)
    plain = profile_layer(dense, z, k_act=4)
    marked = profile_layer(dense, select_positions(z, every), k_act=4)
    assert torch.equal(plain[0], marked[0])
    assert torch.equal(plain[1], marked[1])


# --- 断り方 -----------------------------------------------------------------

def test_a_mask_of_the_wrong_shape_is_refused(layer):
    _, z = layer
    with pytest.raises(ValueError, match='対応していない'):
        select_positions(z, torch.ones(BATCH, SEQ + 1, dtype=torch.bool))


def test_an_empty_mask_is_refused(layer):
    _, z = layer
    with pytest.raises(ValueError, match='1つも無い'):
        select_positions(z, torch.zeros(BATCH, SEQ, dtype=torch.bool))


def test_a_two_dimensional_input_is_refused(layer):
    _, z = layer
    with pytest.raises(ValueError, match=r'\[bsz, seq, hidden\]'):
        select_positions(z.reshape(-1, HIDDEN),
                         torch.ones(BATCH * SEQ, dtype=torch.bool))


# --- 校正セットが印を持たないとき -------------------------------------------

def test_the_converter_refuses_a_calibration_set_without_marks():
    from cmoe.assemble import Converter
    from cmoe.data.base import TokenSet

    converter = Converter.__new__(Converter)
    converter.scored_positions_only = True
    plain = TokenSet('wikitext2-train-carve', torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match='印を持たない'):
        converter._profile_mask(plain)


def test_the_converter_does_not_look_for_marks_when_it_is_not_asked():
    from cmoe.assemble import Converter
    from cmoe.data.base import TokenSet

    converter = Converter.__new__(Converter)
    converter.scored_positions_only = False
    plain = TokenSet('wikitext2-train-carve', torch.zeros(2, 4, dtype=torch.long))
    assert converter._profile_mask(plain) is None


# --- 組み立て役まで通す -----------------------------------------------------

N_EXPERTS, N_ACTIVE, SEQLEN = 4, 4, 16


def make_adapter():
    """毎回新しく作る。変換は破壊的なので、1モデルを2通りには変換できない。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    from cmoe.adapters.llama import LlamaAdapter

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN * 8)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.eval()
    model.config.use_cache = False
    return LlamaAdapter(model, seqlen=SEQLEN, device='cpu')


def marked_token_set(segments):
    from cmoe.data.base import TokenSet

    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 128, segments.shape, generator=generator)
    return TokenSet('marked-calib', ids, (), (), segments)


def convert(token_set, scored_only):
    adapter = make_adapter()
    from cmoe.alloc.base import Allocation
    from cmoe.assemble import Converter
    from cmoe.carve.registry import create_carver
    from cmoe.router.registry import create_method

    converter = Converter(
        adapter, create_carver('cmoe', N_EXPERTS), [create_method('cmoe')],
        n_experts=N_EXPERTS, scored_positions_only=scored_only)
    allocation = Allocation(tuple([1] * adapter.n_layers), name='uniform1',
                            n_active_total=N_ACTIVE)
    report = converter.convert(token_set, allocation)
    return [row.baseline_representatives for row in report.layers]


def test_marking_every_position_converts_to_the_same_model():
    """全位置に印を付けたら、印を見ない経路と1ビットも違ってはならない。"""
    from cmoe.data.base import SCORED

    segments = torch.full((2, SEQLEN), SCORED, dtype=torch.int8)
    token_set = marked_token_set(segments)
    plain = convert(token_set, scored_only=False)
    marked = convert(token_set, scored_only=True)
    assert plain == marked


def test_the_marks_reach_the_split():
    """印を絞れば分割は動く。動かなければ印がどこかで落ちている。"""
    from cmoe.data.base import CONTEXT, SCORED

    segments = torch.full((2, SEQLEN), CONTEXT, dtype=torch.int8)
    segments[:, -4:] = SCORED
    token_set = marked_token_set(segments)
    assert convert(token_set, False) != convert(token_set, True)


def test_right_padding_does_not_move_the_real_tokens():
    """右詰めの埋めは実トークンの活性に効かない — 因果 attention だから。

    ここが成り立たないなら、1問1系列のセットは attention マスクを通さなければ
    ならない。``benchqa`` が埋めを右端に置いているのはこの性質に乗っている。
    """
    adapter = make_adapter()
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 128, (1, SEQLEN // 2), generator=generator)
    padded = torch.cat([ids, torch.zeros(1, SEQLEN // 2, dtype=torch.long)], dim=1)

    bare = adapter.capture_layer_inputs(ids)
    with_pad = adapter.capture_layer_inputs(padded)
    z_bare, _ = adapter.forward_attention(
        0, bare.hidden, bare.attention_mask, bare.position_ids)
    z_pad, _ = adapter.forward_attention(
        0, with_pad.hidden, with_pad.attention_mask, with_pad.position_ids)
    assert torch.equal(z_bare, z_pad[:, :SEQLEN // 2])
