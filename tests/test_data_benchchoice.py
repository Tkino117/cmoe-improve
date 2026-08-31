"""選択肢ごとに1系列で引く校正セット（``cmoe.data.benchchoice``）。

``benchqa`` の拡張なので、見るのは足したところだけである。

1. **予算の数え方**が2つあること。既定はタスクごとの問題数で、
   ``gold_scored`` を選ぶと ``benchqa`` とまったく同じ問題が選ばれる
   （これが崩れると report/14 の測定と校正の問題集合が揃わない）
2. **行と問題の対応**が、タスクをまたいでずれないか
3. **切り方が正解肢で ``render_parts`` と一致する**か。校正と評価が同じ切れ目を
   見る、という約束が2実装に割れていないこと

実物のタスクは ``CMOE_BENCHTRAIN_INTEGRATION=1`` のときだけ引く。
"""

import os

import pytest
import torch

from cmoe.data import benchchoice, benchqa
from cmoe.data.base import SCORED
from cmoe.data.benchtrain import TASKS
from cmoe.data.registry import CALIBRATION_SETS

INTEGRATION = os.environ.get('CMOE_BENCHTRAIN_INTEGRATION') == '1'


class FakeTokenizer:
    """1文字1トークン。id は文字コード。先頭に BOS(=1) を足す。"""

    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        ids = [1] + [ord(character) for character in text]
        return type('Encoded', (), {
            'input_ids': torch.tensor([ids], dtype=torch.long)})()


def fake_gold_parts(count, context_size=6, continuation_size=3):
    """``benchqa`` が読む形。1問1本、正解肢だけ。"""
    return tuple((chr(ord('a') + index % 20) * context_size,
                  chr(ord('A') + index % 20) * continuation_size)
                 for index in range(count))


def fake_choice_pool(count, n_choices=3, context_size=6, continuation_size=3):
    """``benchchoice`` が読む形。正解肢は ``fake_gold_parts`` と同じ長さにする。

    正解を選択肢0 に置き、不正解は続きの長さを変えて並べる。正解肢の採点位置の
    数だけが予算に効くので、不正解の長さは引きに影響してはいけない。
    """
    pool = []
    for context, continuation in fake_gold_parts(count, context_size,
                                                 continuation_size):
        choices = [(context, continuation)]
        for other in range(1, n_choices):
            choices.append((context, chr(ord('Z') - other) * (
                continuation_size + other)))
        pool.append((tuple(choices), 0))
    return tuple(pool)


# --- 1. 同じ引数なら同じ問題 -------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 7])
def test_the_same_questions_are_drawn_as_benchqa(seed):
    """予算は**正解肢の**採点位置で数える。不正解肢の長さは引きを動かさない。

    ここが揃っているので、report/14 と校正の問題集合を揃えたまま目的関数だけを
    替えられる。
    """
    tokenizer = FakeTokenizer()
    _, _, gold_component = benchqa.draw_component(
        tokenizer, fake_gold_parts(40), budget=20, seed=seed, name='piqa')
    _, _, _, component = benchchoice.draw_component(
        tokenizer, fake_choice_pool(40), budget=20, seed=seed, name='piqa',
        budget_unit='gold_scored')
    assert component.documents == gold_component.documents
    assert component.gold_scored_tokens == gold_component.scored_tokens


def test_the_number_of_choices_does_not_change_the_questions():
    tokenizer = FakeTokenizer()
    two = benchchoice.draw_component(
        tokenizer, fake_choice_pool(40, n_choices=2), 20, 0, 'piqa',
        budget_unit='gold_scored')[3]
    five = benchchoice.draw_component(
        tokenizer, fake_choice_pool(40, n_choices=5), 20, 0, 'piqa',
        budget_unit='gold_scored')[3]
    assert two.documents == five.documents
    assert two.n_rows * 5 == five.n_rows * 2


def test_a_pool_that_cannot_reach_the_budget_is_refused():
    with pytest.raises(ValueError, match='届かない'):
        benchchoice.draw_component(FakeTokenizer(), fake_choice_pool(3), 100,
                                   0, 'piqa')


# --- 2. 行と問題の対応 -------------------------------------------------------

def test_every_choice_of_a_question_gets_its_own_row():
    rows, marks, groups, component = benchchoice.draw_component(
        FakeTokenizer(), fake_choice_pool(40, n_choices=3), 20, 0, 'piqa')
    assert len(rows) == len(marks) == component.n_rows
    assert component.n_rows == 3 * len(component.documents)
    seen = sorted(row for choices, _ in groups for row in choices)
    assert seen == list(range(len(rows)))
    assert all(gold == 0 for _, gold in groups)


def test_the_marks_are_on_every_choice_not_only_the_gold():
    """不正解肢にも印が立つ。マージンは K 本すべての点を要る。"""
    rows, marks, groups, _ = benchchoice.draw_component(
        FakeTokenizer(), fake_choice_pool(10, n_choices=3), 6, 0, 'piqa')
    for choices, _ in groups:
        for row in choices:
            assert int((marks[row] == SCORED).sum()) > 0


def test_rows_are_shifted_by_the_tasks_already_packed(monkeypatch):
    """タスクごとに 0 から数えた行番号を、積んだぶんだけずらしているか。

    ここを忘れると、2つ目以降のタスクの問題が1つ目のタスクの行を指す — 別の
    問題の選択肢を1問として比べる形の壊れ方になる。
    """
    monkeypatch.setattr(benchchoice, 'load_tokenizer',
                        lambda model: FakeTokenizer())
    monkeypatch.setattr(benchchoice, 'documents',
                        lambda task, cache_dir=None: fake_choice_pool(40))
    token_set = benchchoice.calibration('fake-model', 10, 2, seed=0,
                                        tasks=('piqa', 'winogrande'))
    groups = token_set.choices
    groups.check_rows(token_set.input_ids.shape[0])
    assert set(groups.tasks) == {'piqa', 'winogrande'}
    # 1つ目のタスクの最後の行より、2つ目のタスクの最初の行が後にある
    first = [rows for rows, task in zip(groups.rows, groups.tasks)
             if task == 'piqa']
    second = [rows for rows, task in zip(groups.rows, groups.tasks)
              if task == 'winogrande']
    assert max(max(rows) for rows in first) < min(min(rows) for rows in second)


def test_the_set_is_registered_under_its_own_name():
    assert 'benchchoice' in CALIBRATION_SETS
    for task in TASKS:
        assert f'benchchoice:{task}' in CALIBRATION_SETS
    assert 'benchchoice:arc' in CALIBRATION_SETS


# --- 3. 切り方が正解肢で render_parts と一致する -------------------------------

@pytest.mark.skipif(not INTEGRATION, reason='データセットの取得が要る')
def test_the_gold_choice_is_exactly_what_render_parts_returns():
    """2実装に割れていないこと。``render_parts`` はこの gold 番目そのものである。"""
    from cmoe.data.harness import (calibration_docs, render_choice_parts,
                                   render_parts)

    for index, (task, doc) in enumerate(calibration_docs('piqa')):
        if index >= 20:
            break
        parts, gold = render_choice_parts(task, doc)
        assert parts[gold] == render_parts(task, doc)


@pytest.mark.skipif(not INTEGRATION, reason='データセットの取得が要る')
def test_winogrande_puts_the_choices_in_the_context():
    """``multiple_input`` のタスクは選択肢が文脈側に立ち、続きが共通になる。"""
    from cmoe.data.harness import calibration_docs, render_choice_parts

    task, doc = next(iter(calibration_docs('winogrande')))
    parts, _ = render_choice_parts(task, doc)
    assert len({context for context, _ in parts}) == len(parts)
    assert len({continuation for _, continuation in parts}) == 1


@pytest.mark.skipif(not INTEGRATION, reason='データセットの取得が要る')
def test_draws_the_same_questions_as_benchqa_on_the_real_tasks():
    model = 'meta-llama/Llama-2-7b-hf'
    plain = benchqa.calibration(model, 64, 5, seed=0)
    choice = benchchoice.calibration(model, 64, 5, seed=0,
                                     budget_unit='gold_scored')
    assert ([component.documents for component in choice.components]
            == [component.documents for component in plain.components])
    assert choice.choices.n_questions == sum(
        len(component.documents) for component in plain.components)
    assert choice.input_ids.shape[0] > plain.input_ids.shape[0]
    choice.choices.check_rows(choice.input_ids.shape[0])


# --- 予算の単位 --------------------------------------------------------------

def test_the_question_budget_draws_that_many_questions():
    """既定の数え方。続きの長さが問題数を決めてはいけない。"""
    tokenizer = FakeTokenizer()
    short = benchchoice.draw_component(
        tokenizer, fake_choice_pool(40, continuation_size=2), 12, 0, 'piqa')[3]
    long = benchchoice.draw_component(
        tokenizer, fake_choice_pool(40, continuation_size=9), 12, 0, 'piqa')[3]
    assert len(short.documents) == len(long.documents) == 12
    # 位置で数えていれば、続きが長いほうが少ない問題で予算に届いてしまう
    assert long.gold_scored_tokens > short.gold_scored_tokens


def test_the_two_budget_units_disagree_when_the_continuations_differ():
    """揃え方の違いが実際に出ること。出ないなら片方を持つ意味が無い。"""
    tokenizer = FakeTokenizer()
    positions = benchchoice.draw_component(
        tokenizer, fake_choice_pool(60, continuation_size=9), 36, 0, 'piqa',
        budget_unit='gold_scored')[3]
    questions = benchchoice.draw_component(
        tokenizer, fake_choice_pool(60, continuation_size=9), 36, 0, 'piqa')[3]
    assert len(positions.documents) < len(questions.documents)


def test_an_unknown_budget_unit_is_refused():
    with pytest.raises(ValueError, match='予算の単位'):
        benchchoice.draw_component(FakeTokenizer(), fake_choice_pool(10), 2, 0,
                                   'piqa', budget_unit='tokens')


def test_the_benchqa_shaped_set_is_registered_separately():
    assert 'benchchoice-qa' in CALIBRATION_SETS
