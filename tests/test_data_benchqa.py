"""1問1系列の校正セット（``cmoe.data.benchqa``）。

CPU だけで見るのは4つ — 切れ目の取り方（``encode_pair``）、印の位置
（``build_row``）、引き方（``draw_component``）、右詰め（``pack``）である。
実物のタスクは使わない（データセットの取得が要る）。実物を引くところは
``CMOE_BENCHTRAIN_INTEGRATION=1`` のときだけ走らせる。
"""

import os

import pytest
import torch

from cmoe.data.base import CONTEXT, PAD, SCORED
from cmoe.data.benchqa import (MAX_TOKENS, build_row, draw_component,
                               encode_pair, pack, pad_token_id)
from cmoe.data.benchtrain import TASKS
from cmoe.data.registry import CALIBRATION_SETS

INTEGRATION = os.environ.get('CMOE_BENCHTRAIN_INTEGRATION') == '1'


class FakeTokenizer:
    """1文字1トークン。id は文字コード。先頭に BOS(=1) を足す。

    実物と同じ性質を2つだけ持たせてある — 先頭に特殊トークンが付くことと、
    連結してからトークナイズすると繋ぎ目が別のトークンになりうること
    （ここでは起きないが、``encode_pair`` は起きても正しい切れ目を出す）。
    """

    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        ids = [1] + [ord(character) for character in text]
        return type('Encoded', (), {
            'input_ids': torch.tensor([ids], dtype=torch.long)})()


def fake_parts(count, context_size=6, continuation_size=3):
    """(文脈, 続き) の対を count 個。文脈は問題ごとに違う文字にする。"""
    return tuple((chr(ord('a') + index % 20) * context_size,
                  chr(ord('A') + index % 20) * continuation_size)
                 for index in range(count))


# --- 切れ目の取り方 ---------------------------------------------------------

def test_the_boundary_is_the_length_of_the_context_alone():
    ids, boundary = encode_pair(FakeTokenizer(), 'abc', 'XY')
    # BOS + 'abc' + 'XY'
    assert ids.tolist() == [1, ord('a'), ord('b'), ord('c'), ord('X'), ord('Y')]
    assert boundary == 4


def test_a_trailing_space_moves_to_the_continuation():
    """lm-eval と同じ扱い。単語境界のトークン化を保つための処理である。"""
    ids, boundary = encode_pair(FakeTokenizer(), 'abc ', 'XY')
    assert ids.tolist() == [1, ord('a'), ord('b'), ord('c'), ord(' '),
                            ord('X'), ord('Y')]
    # 空白は続き側へ移るので、文脈は 'abc' の 4 トークンで切れる
    assert boundary == 4


def test_an_empty_context_is_refused():
    with pytest.raises(ValueError, match='文脈が空'):
        encode_pair(FakeTokenizer(), '', 'XY')


# --- 印の位置 ---------------------------------------------------------------

def test_the_marks_sit_one_position_before_the_continuation():
    """位置 i の活性が作るのはトークン i+1 の対数確率なので、1つ手前に立つ。"""
    ids = torch.arange(6)
    row, marks, truncated = build_row(ids, boundary=4)
    assert not truncated
    assert marks.tolist() == [CONTEXT, CONTEXT, CONTEXT, SCORED, SCORED, CONTEXT]


def test_there_are_as_many_marks_as_continuation_tokens():
    for length, boundary in ((6, 4), (10, 2), (3, 1), (20, 19)):
        _, marks, _ = build_row(torch.arange(length), boundary)
        assert int((marks == SCORED).sum()) == length - boundary


def test_the_last_position_is_never_marked():
    """最後の位置が出す予測は、系列の外のトークンに向いている。"""
    _, marks, _ = build_row(torch.arange(6), boundary=1)
    assert marks[-1] == CONTEXT


def test_a_question_without_a_continuation_is_refused():
    with pytest.raises(ValueError, match='続きの開始位置'):
        build_row(torch.arange(6), boundary=6)


def test_a_question_without_a_context_is_refused():
    with pytest.raises(ValueError, match='続きの開始位置'):
        build_row(torch.arange(6), boundary=0)


def test_a_long_question_keeps_its_first_token_and_all_its_marks():
    ids = torch.arange(100)
    row, marks, truncated = build_row(ids, boundary=90, max_tokens=20)
    assert truncated
    assert row.shape[0] == 20
    # 先頭（BOS の居場所）は残り、落ちるのはその後ろ
    assert row[0] == 0
    assert row[1] == 81
    # 続きは 10 トークンあるので、印も 10 立ったまま
    assert int((marks == SCORED).sum()) == 10


def test_truncating_away_the_whole_context_is_refused():
    with pytest.raises(ValueError, match='文脈が残らない'):
        build_row(torch.arange(100), boundary=5, max_tokens=20)


# --- 引き方 -----------------------------------------------------------------

def test_it_draws_until_the_budget_is_reached():
    rows, marks, component = draw_component(
        FakeTokenizer(), fake_parts(50), budget=12, seed=0, name='fake')
    # 1問につき印は3つ（続きが3文字）なので、12 にちょうど届く4問で止まる
    assert len(rows) == len(marks) == 4
    assert component.scored_tokens == 12


def test_it_does_not_cut_the_question_that_crosses_the_budget():
    _, _, component = draw_component(
        FakeTokenizer(), fake_parts(50), budget=10, seed=0, name='fake')
    assert component.scored_tokens == 12
    assert len(component.documents) == 4


def test_the_same_seed_draws_the_same_questions():
    first = draw_component(FakeTokenizer(), fake_parts(50), 12, 0, 'fake')[2]
    again = draw_component(FakeTokenizer(), fake_parts(50), 12, 0, 'fake')[2]
    other = draw_component(FakeTokenizer(), fake_parts(50), 12, 1, 'fake')[2]
    assert first.documents == again.documents
    assert first.documents != other.documents


def test_tasks_with_different_names_draw_differently():
    """benchtrain と同じ規則。同じ seed を配ってもタスクごとに別の並びになる。"""
    first = draw_component(FakeTokenizer(), fake_parts(50), 12, 0, 'one')[2]
    second = draw_component(FakeTokenizer(), fake_parts(50), 12, 0, 'two')[2]
    assert first.documents != second.documents


def test_a_pool_too_small_is_refused_instead_of_reusing_questions():
    with pytest.raises(ValueError, match='届かない'):
        draw_component(FakeTokenizer(), fake_parts(2), budget=100, seed=0,
                       name='fake')


def test_the_component_records_what_it_drew():
    _, _, component = draw_component(
        FakeTokenizer(), fake_parts(50), 12, 0, 'fake')
    assert component.metadata() == {
        'name': 'fake',
        'n_questions': 4,
        'documents': list(component.documents),
        'pool_documents': 50,
        'scored_tokens': 12,
        # BOS + 文脈6文字 が4問。印の付かない位置がそのぶん残る
        'context_tokens': 4 * (1 + 6),
        'truncated': 0,
        'split': 'train',
    }


# --- 右詰め -----------------------------------------------------------------

def test_padding_goes_on_the_right_and_is_never_marked():
    rows = [torch.arange(3), torch.arange(5)]
    marks = [torch.full((3,), CONTEXT, dtype=torch.int8),
             torch.full((5,), SCORED, dtype=torch.int8)]
    input_ids, segments = pack(rows, marks, pad_id=7)
    assert input_ids.shape == (2, 5)
    assert input_ids[0].tolist() == [0, 1, 2, 7, 7]
    assert segments[0].tolist() == [CONTEXT, CONTEXT, CONTEXT, PAD, PAD]
    assert segments[1].tolist() == [SCORED] * 5


def test_the_pad_id_falls_back_to_the_end_of_text():
    assert pad_token_id(FakeTokenizer()) == 2


# --- 登録 -------------------------------------------------------------------

def test_every_task_has_a_single_task_calibration_set():
    for task in TASKS:
        assert f'benchqa:{task}' in CALIBRATION_SETS
    assert 'benchqa' in CALIBRATION_SETS
    assert 'benchqa:arc' in CALIBRATION_SETS


# --- 実物を引く -------------------------------------------------------------

@pytest.mark.skipif(not INTEGRATION, reason='データセットの取得が要る')
def test_draws_a_stratified_set():
    from cmoe.data import benchqa

    # 採点位置 5×64 = 320。タスクへ 64 ずつ配られる
    token_set = benchqa.calibration('meta-llama/Llama-2-7b-hf', 64, 5, seed=0)
    assert token_set.segments.shape == token_set.input_ids.shape
    assert token_set.input_ids.shape[1] <= MAX_TOKENS
    assert [component.name for component in token_set.components] == list(TASKS)
    for component in token_set.components:
        assert component.scored_tokens >= 64
    assert int((token_set.segments == SCORED).sum()) == sum(
        component.scored_tokens for component in token_set.components)
