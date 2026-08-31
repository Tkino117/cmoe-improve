"""選択問題の採点そのものを目的関数にするオラクル（``cmoe.alloc.oracles.margin``）。

CPU で見るのは4つある。

1. **束ね方の対応**（``build_choice_scoring``）— 印の位置が予測すべきトークンを
   1つ後ろから拾えているか、行と問題の対応が崩れていないか
2. **マージンの定義**が、評価側に既にある指標と両端で一致するか。β → ∞ で
   ``bench_stats.margin`` の符号反転、β = 1 の softplus で ``bench_stats.gold_nll``。
   これが一致しないなら、校正と評価が別の量を見ていることになる
3. **系列の対数尤度**が lm-eval と同じ和になっているか（採点位置の1列の和）
4. **極小モデルで最後まで走る**か。数値は見ない

読み出しは作らず、``score_logits`` に直接 logits を渡す形で 1〜3 を閉じる。
語彙4・位置10 の手で追える大きさで、マージンの定義そのものを見る。
"""

import math

import pytest
import torch

from cmoe.adapters.llama import LlamaAdapter
from cmoe.alloc.base import build_choice_scoring
from cmoe.alloc.oracles.base import LayerWalk
from cmoe.alloc.oracles.margin import (MarginOracle, apply_loss, macro_mean,
                                       margins, sequence_loglikelihoods)
from cmoe.alloc.oracles.registry import create_oracle
from cmoe.assemble import layer_factory
from cmoe.carve.registry import create_carver
from cmoe.data.base import CONTEXT, PAD, SCORED, ChoiceGroups, TokenSet
from cmoe.eval import bench_stats

N_EXPERTS, N_ACTIVE, SEQLEN = 4, 3, 16


# --- 手で作る小さな校正セット -------------------------------------------------

def tiny_token_set():
    """2問。問題0 は選択肢2本、問題1 は3本。採点位置は選択肢ごとに2つ。

    幅は6。各行は [BOS, 文脈, 文脈, 続き, 続き, 埋め] の形で、印は続きの
    トークンより1つ手前の位置2..3 に立つ。
    """
    rows = [
        [1, 10, 11, 20, 21, 0],   # 問題0 / 選択肢0（正解）
        [1, 10, 11, 30, 31, 0],   # 問題0 / 選択肢1
        [1, 12, 13, 40, 41, 0],   # 問題1 / 選択肢0
        [1, 12, 13, 50, 51, 0],   # 問題1 / 選択肢1（正解）
        [1, 12, 13, 60, 61, 0],   # 問題1 / 選択肢2
    ]
    segments = torch.full((5, 6), CONTEXT, dtype=torch.int8)
    segments[:, 2:4] = SCORED
    segments[:, 4:] = PAD
    choices = ChoiceGroups(rows=((0, 1), (2, 3, 4)), gold=(0, 1),
                           tasks=('taskA', 'taskB'))
    return TokenSet('tiny', torch.tensor(rows, dtype=torch.long), (), (),
                    segments, choices)


def tiny_scoring():
    tokens = tiny_token_set()
    return build_choice_scoring(
        tokens.input_ids, tokens.scored_mask(), tokens.choices.rows,
        tokens.choices.gold, tokens.choices.tasks)


# --- 1. 束ね方の対応 ---------------------------------------------------------

def test_the_target_of_a_scored_position_is_the_token_one_step_later():
    """印は続きのトークンより1つ手前に立つ。拾うのはその1つ後ろである。

    ここがずれると、校正は「文脈の最後の1トークン」を採点していることになる。
    """
    scoring = tiny_scoring()
    # 行0 の印は位置2,3 で、予測すべきは位置3,4 のトークン = 20, 21
    assert scoring.targets[:2].tolist() == [20, 21]
    assert scoring.targets[2:4].tolist() == [30, 31]
    assert scoring.n_positions == 10


def test_the_positions_are_listed_row_by_row():
    """並びは行優先。``forward_suffix(keep=...)`` が返す順そのものである。"""
    scoring = tiny_scoring()
    assert scoring.row_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_the_rows_of_a_question_are_padded_to_a_rectangle():
    """K は問題ごとに違ってよい。埋めた升は不正解にも正解にも数えない。"""
    scoring = tiny_scoring()
    assert scoring.choice_rows.tolist() == [[0, 1, 0], [2, 3, 4]]
    assert scoring.choice_mask.tolist() == [[True, True, False],
                                            [True, True, True]]
    assert scoring.gold_column.tolist() == [0, 1]


def test_a_mark_on_the_last_column_is_refused():
    """最終列の印が予測するトークンは系列の外にある。黙って隣を読ませない。"""
    tokens = tiny_token_set()
    segments = tokens.segments.clone()
    segments[0, -1] = SCORED
    with pytest.raises(ValueError, match='最終列'):
        build_choice_scoring(tokens.input_ids, segments == SCORED,
                             tokens.choices.rows, tokens.choices.gold,
                             tokens.choices.tasks)


def test_rows_that_do_not_cover_every_sequence_are_refused():
    """別の問題の選択肢が1本の問題に混ざる形の壊れ方を、作った直後に閉じる。"""
    with pytest.raises(ValueError, match='1対1'):
        ChoiceGroups(rows=((0, 1), (2, 3)), gold=(0, 1),
                     tasks=('a', 'b')).check_rows(5)


def test_a_question_with_one_choice_is_refused():
    with pytest.raises(ValueError, match='比べる相手'):
        ChoiceGroups(rows=((0,),), gold=(0,), tasks=('a',))


# --- 2. 系列の対数尤度 -------------------------------------------------------

def logits_for(values):
    """[位置, 語彙] を作る。``values[p][v]`` がそのまま logit になる。"""
    return torch.tensor(values, dtype=torch.float32)


def test_a_sequence_score_is_the_sum_over_its_scored_positions():
    """lm-eval が選択肢に付ける点そのもの。長さで割らない生の和である。"""
    scoring = tiny_scoring()
    vocab = 64
    # 位置ごとに、拾うべきトークンだけ logit を大きくする（値は位置で変える）
    logits = torch.zeros((scoring.n_positions, vocab))
    for position, (target, offset) in enumerate(
            zip(scoring.targets.tolist(), range(scoring.n_positions))):
        logits[position, target] = float(offset)
    row_ll = sequence_loglikelihoods(logits, scoring)

    expected = torch.zeros(5, dtype=torch.float64)
    for position in range(scoring.n_positions):
        log_p = logits[position].log_softmax(dim=-1)
        expected[scoring.row_index[position]] += log_p[scoring.targets[position]]
    assert torch.allclose(row_ll, expected, atol=1e-9)


def test_the_readout_must_match_the_number_of_marks():
    scoring = tiny_scoring()
    with pytest.raises(ValueError, match='対応していない'):
        sequence_loglikelihoods(torch.zeros((3, 8)), scoring)


def test_splitting_the_positions_does_not_change_the_sequence_scores():
    scoring = tiny_scoring()
    torch.manual_seed(0)
    logits = torch.randn(scoring.n_positions, 64)
    whole = sequence_loglikelihoods(logits, scoring, token_chunk=1024)
    split = sequence_loglikelihoods(logits, scoring, token_chunk=3)
    assert torch.allclose(whole, split, atol=1e-12)


# --- 3. マージンの定義が、評価側の指標と両端で一致する -------------------------

def reference_row_loglikelihoods():
    """手で置いた選択肢ごとの対数尤度。問題0 は正解が勝ち、問題1 は負ける。"""
    return torch.tensor([-3.0, -5.0,           # 問題0: 正解(-3) > 不正解(-5)
                         -2.0, -4.0, -9.0],    # 問題1: 正解(-4) < 不正解(-2)
                        dtype=torch.float64)


def bench_samples():
    """同じ数を ``bench_stats`` が読む形にしたもの。"""
    row_ll = reference_row_loglikelihoods().tolist()
    return type('Samples', (), {
        'loglikelihoods': [row_ll[0:2], row_ll[2:5]],
        'gold': [0, 1],
    })()


def test_a_sharp_softmin_is_the_benchmark_margin_with_the_sign_flipped():
    """β → ∞ の極限。符号が ``acc`` そのものになる側の端である。"""
    scoring = tiny_scoring()
    values = margins(reference_row_loglikelihoods(), scoring, beta=1e4)
    expected = [-value for value in bench_stats.margin(bench_samples())]
    assert values.tolist() == pytest.approx(expected, abs=1e-6)


def test_softplus_at_beta_one_is_the_benchmark_gold_nll():
    """β = 1 に softplus を掛けると、選択肢上の交差エントロピーと厳密に一致する。

    M = log( Σ_不正解 p / p_正解 ) なので、log(1 + exp(M)) が
    −log( p_正解 / Σ_全部 p ) になる。校正の目的関数と評価指標が同じ族から
    出ていることの、いちばん強い形の確認である。
    """
    scoring = tiny_scoring()
    values = apply_loss(margins(reference_row_loglikelihoods(), scoring,
                                beta=1.0), 'softplus')
    expected = bench_stats.gold_nll(bench_samples())
    assert values.tolist() == pytest.approx(expected, abs=1e-9)


def test_the_sign_of_the_margin_is_whether_the_question_is_wrong():
    """M < 0 が正解。``acc`` の符号そのものである。"""
    scoring = tiny_scoring()
    values = margins(reference_row_loglikelihoods(), scoring, beta=1e4)
    correct = [1.0 if value < 0 else 0.0 for value in values.tolist()]
    assert correct == bench_stats.accuracy(bench_samples())


def test_a_softer_softmin_lets_the_runner_up_count():
    """β を下げると2位以下の不正解も効く。素の min との違いはそこにしかない。

    問題1 は不正解が2本（-2 と -9）ある。硬い min は -2 しか見ないので、
    -9 を -2.5 に上げても値が動かないが、柔らかい softmin では動く。
    """
    scoring = tiny_scoring()
    base = reference_row_loglikelihoods()
    lifted = base.clone()
    lifted[4] = -2.5
    hard = (margins(base, scoring, beta=1e4)[1],
            margins(lifted, scoring, beta=1e4)[1])
    soft = (margins(base, scoring, beta=1.0)[1],
            margins(lifted, scoring, beta=1.0)[1])
    assert hard[0].item() == pytest.approx(hard[1].item(), abs=1e-6)
    assert soft[1].item() > soft[0].item() + 1e-3


def test_padded_choices_never_count_as_distractors():
    """問題0 は選択肢2本しか無い。埋めた3列目が不正解として効いてはいけない。

    ``choice_rows`` の埋めは 0 なので、埋めを数えると問題0 の正解肢（行0）が
    自分の不正解として立つ。手で置いた2択の値と突き合わせて閉じる:
    正解 -3、不正解 -5 なので M = 3 + (-5) = -2 でなければならない
    （埋めを数えていれば logsumexp(-5, -3) = -2.873 が入り、+0.127 になる）。
    """
    scoring = tiny_scoring()
    value = margins(reference_row_loglikelihoods(), scoring, beta=1.0)[0]
    assert value.item() == pytest.approx(-2.0, abs=1e-12)


def test_beta_must_be_positive():
    with pytest.raises(ValueError, match='β'):
        margins(reference_row_loglikelihoods(), tiny_scoring(), beta=0.0)


# --- 束ね方（タスクを等しく） -------------------------------------------------

def test_tasks_are_weighted_equally_not_questions():
    """ベンチの集計と同じマクロ平均。問題数がタスクで揃わないので効く。"""
    scoring = tiny_scoring()
    # 問題0 は taskA、問題1 は taskB。1問ずつなので単純平均と一致する
    values = torch.tensor([1.0, 3.0], dtype=torch.float64)
    macro, per_task = macro_mean(values, scoring)
    assert macro == pytest.approx(2.0)
    assert per_task.tolist() == pytest.approx([1.0, 3.0])


# --- 4. 極小モデルで最後まで走る ----------------------------------------------

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


def make_walk(adapter, tokens=None):
    tokens = tokens or tiny_token_set()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    factory = layer_factory(create_carver('cmoe', N_EXPERTS), N_EXPERTS,
                            device=adapter.device)
    scoring = None
    if tokens.choices is not None:
        scoring = build_choice_scoring(
            tokens.input_ids, tokens.scored_mask(), tokens.choices.rows,
            tokens.choices.gold, tokens.choices.tasks)
    return LayerWalk(adapter, inputs, factory, N_EXPERTS,
                     n_active_total=N_ACTIVE, choices=scoring)


def test_the_oracle_scores_a_candidate_end_to_end(adapter):
    walk = make_walk(adapter)
    oracle = create_oracle('margin', walk)
    state = walk.root()
    profile = walk.profile(state, 0)
    carved = walk.carve(profile, 1)
    child = walk.propagate(profile, carved)
    result = oracle.measure(profile, carved, state, child)

    assert math.isfinite(result.score)
    assert result.details['n_questions'] == 2
    assert set(result.details['per_task']) == {'taskA', 'taskB'}
    # 自分の層と、そのあと dense で走らせた層
    assert result.cost == float(walk.n_layers)
    assert result.cost_unit == 'layer_forwards'


def test_the_dense_reference_is_reproducible(adapter):
    """同じ dense を2回測った差が、CPU では厳密に 0 になる。"""
    oracle = create_oracle('margin', make_walk(adapter))
    assert oracle.nondeterminism_floor() == pytest.approx(0.0, abs=1e-12)


def test_the_calibration_accuracy_is_the_sign_of_the_margin(adapter):
    """内訳に出る ``calibration_acc`` は追加の forward 無しで出る量である。"""
    walk = make_walk(adapter)
    oracle = create_oracle('margin', walk)
    state = walk.root()
    logits = oracle.run_suffix(0, state.hidden)
    score, details = oracle.score_logits(logits)
    assert 0.0 <= details['calibration_acc'] <= 1.0
    assert details['margin'] == score


def test_a_calibration_set_without_distractors_is_refused(adapter):
    """benchqa は正解肢しか持たない。比べる相手が無いことを先に断る。"""
    tokens = TokenSet('plain', torch.randint(0, 128, (2, SEQLEN)))
    with pytest.raises(ValueError, match='benchchoice'):
        create_oracle('margin', make_walk(adapter, tokens))


@pytest.mark.parametrize('loss', ['raw', 'softplus'])
def test_both_losses_run(adapter, loss):
    oracle = create_oracle('margin', make_walk(adapter), loss=loss)
    assert math.isfinite(oracle.reference_score())


def test_an_unknown_loss_is_refused(adapter):
    with pytest.raises(ValueError, match='未知の損失'):
        create_oracle('margin', make_walk(adapter), loss='hinge')


def test_the_marks_of_the_walk_must_match_the_calibration_input(adapter):
    """印と校正入力の形が食い違ったまま走らせない。"""
    tokens = tiny_token_set()
    inputs = adapter.capture_layer_inputs(tokens.input_ids)
    factory = layer_factory(create_carver('cmoe', N_EXPERTS), N_EXPERTS,
                            device=adapter.device)
    other = tiny_token_set()
    scoring = build_choice_scoring(
        other.input_ids[:, :5], other.scored_mask()[:, :5],
        other.choices.rows, other.choices.gold, other.choices.tasks)
    with pytest.raises(ValueError, match='対応していない'):
        LayerWalk(adapter, inputs, factory, N_EXPERTS,
                  n_active_total=N_ACTIVE, choices=scoring)
