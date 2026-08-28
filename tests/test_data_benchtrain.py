"""ベンチマークの train split から作る校正セット。

CPU だけで見るのは3つ — 本数の配り方（``allocate``）、1問をテキストに直す形
（``harness.render_document``）、引き方（``draw_component``）である。実物の
タスクは使わない（データセットの取得が要る）。実物を引くところは
``CMOE_BENCHTRAIN_INTEGRATION=1`` のときだけ走らせる。
"""

import os
import random
import types
from types import SimpleNamespace

import pytest
import torch

from cmoe.data.benchtrain import (MARGIN, MIN_MARGIN, TASKS, allocate,
                                  draw_component)
from cmoe.data.harness import render_document
from cmoe.data.wikitext2 import intervals_overlap
from cmoe.eval import bench

INTEGRATION = os.environ.get('CMOE_BENCHTRAIN_INTEGRATION') == '1'


class FakeTokenizer:
    """1トークン = chars_per_token 文字。トークン id は連番。"""

    def __init__(self, chars_per_token=4):
        self.chars_per_token = chars_per_token

    def __call__(self, text, return_tensors=None):
        count = len(text) // self.chars_per_token
        return SimpleNamespace(
            input_ids=torch.arange(count, dtype=torch.long).unsqueeze(0))


class FakeTask:
    """``doc_to_*`` と ``target_delimiter`` だけを持つ、タスクの代わり。"""

    def __init__(self, multiple_input=0, delimiter=' '):
        self.config = types.SimpleNamespace(task='fake',
                                            target_delimiter=delimiter)
        self.multiple_input = multiple_input

    def doc_to_choice(self, doc):
        return doc['choices']

    def doc_to_text(self, doc):
        return doc['gold'] if self.multiple_input else doc['text']

    def doc_to_target(self, doc):
        return doc['target'] if self.multiple_input else doc['gold']


def make_texts(count, size, tag='x'):
    return tuple(tag * size for _ in range(count))


# --- 本数の配り方 -----------------------------------------------------------

def test_every_task_gets_at_least_one_window():
    quota = allocate(len(TASKS))
    assert quota == {name: 1 for name in TASKS}


def test_the_remainder_goes_in_task_order():
    """余りは並びの順に1本ずつ。seed には依らない。"""
    quota = allocate(8)
    assert sum(quota.values()) == 8
    assert quota == {'piqa': 2, 'winogrande': 2, 'arc_easy': 2,
                     'arc_challenge': 1, 'hellaswag': 1}


def test_a_single_task_takes_every_window():
    """タスクを1つに絞った系。本数は減らさず、全部をそのタスクから引く。"""
    assert allocate(8, ('arc_challenge',)) == {'arc_challenge': 8}


def test_a_count_below_the_number_of_tasks_is_refused():
    with pytest.raises(ValueError, match='1本ずつ配れない'):
        allocate(len(TASKS) - 1)


def test_the_task_list_matches_the_benchmark():
    """校正が引くタスクと、評価が測るタスクが黙って分かれないための検査。"""
    assert TASKS == bench.DEFAULT_TASKS


def test_every_task_has_a_single_task_calibration_set():
    """5×5 の行になる ``benchtrain:<task>`` が、タスクの数だけ揃っている。"""
    from cmoe.data.registry import CALIBRATION_SETS

    for task in TASKS:
        assert f'benchtrain:{task}' in CALIBRATION_SETS


# --- 1問をテキストに直す ----------------------------------------------------

def test_a_document_is_the_context_and_the_correct_continuation():
    task = FakeTask()
    doc = {'text': 'Question: なぜ\nAnswer:', 'choices': ['正解', '不正解'],
           'gold': 0}
    assert render_document(task, doc) == 'Question: なぜ\nAnswer: 正解'


def test_the_wrong_choices_do_not_appear():
    task = FakeTask()
    doc = {'text': 'Q:', 'choices': ['あたり', 'はずれA', 'はずれB'], 'gold': 0}
    text = render_document(task, doc)
    assert 'はずれA' not in text and 'はずれB' not in text


def test_a_multiple_input_task_puts_the_choice_in_the_context():
    """WinoGrande の形。選択肢が文脈の側に立ち、続きは全選択肢で共通である。"""
    task = FakeTask(multiple_input=2)
    doc = {'choices': ['A は', 'B は'], 'gold': 1, 'target': '嫌いだった。'}
    assert render_document(task, doc) == 'B は 嫌いだった。'


def test_the_delimiter_is_the_one_the_harness_uses():
    task = FakeTask(delimiter='')
    doc = {'text': 'Q:', 'choices': ['A'], 'gold': 0}
    assert render_document(task, doc) == 'Q:A'


# --- 引き方 -----------------------------------------------------------------

def test_windows_within_a_task_do_not_overlap():
    tokenizer = FakeTokenizer()
    texts = make_texts(400, 500)
    _, component = draw_component(tokenizer, texts, 128, 4, 0, 'piqa')
    for left in range(len(component.starts)):
        for right in range(left + 1, len(component.starts)):
            assert not intervals_overlap(
                component.starts[left], component.starts[right], 128)


def test_the_same_seed_draws_the_same_windows():
    tokenizer = FakeTokenizer()
    texts = make_texts(400, 500)
    first, _ = draw_component(tokenizer, texts, 128, 4, 0, 'piqa')
    again, _ = draw_component(tokenizer, texts, 128, 4, 0, 'piqa')
    other, _ = draw_component(tokenizer, texts, 128, 4, 1, 'piqa')
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_tasks_with_different_names_draw_differently():
    """タスクごとに独立の種を使う。同じ母集団でも同じ位置には落ちない。"""
    tokenizer = FakeTokenizer()
    texts = make_texts(400, 500)
    left, _ = draw_component(tokenizer, texts, 128, 4, 0, 'piqa')
    right, _ = draw_component(tokenizer, texts, 128, 4, 0, 'hellaswag')
    assert not torch.equal(left, right)


def test_a_pool_too_small_is_refused_instead_of_packing_tighter():
    tokenizer = FakeTokenizer()
    texts = make_texts(4, 100)
    with pytest.raises(ValueError, match='最低'):
        draw_component(tokenizer, texts, 128, 4, 0, 'arc_challenge')


def test_the_component_records_what_it_drew():
    tokenizer = FakeTokenizer()
    texts = make_texts(400, 500)
    _, component = draw_component(tokenizer, texts, 128, 4, 0, 'piqa')
    meta = component.metadata()
    assert meta['name'] == 'piqa'
    assert meta['n_sequences'] == 4
    assert meta['split'] == 'train'
    assert meta['pool_documents'] == 400
    assert meta['n_documents'] == len(set(component.documents))
    assert meta['pool_tokens'] >= 4 * 128 * MIN_MARGIN


def test_only_as_many_documents_as_the_margin_needs_are_read():
    """窓の総面積の MARGIN 倍で足りたら、そこで採るのをやめる。"""
    tokenizer = FakeTokenizer()
    texts = make_texts(4000, 500)
    _, component = draw_component(tokenizer, texts, 128, 2, 0, 'piqa')
    assert component.n_chars < 2 * 128 * 4 * MARGIN + 500


# --- 実物 -------------------------------------------------------------------

@pytest.mark.skipif(not INTEGRATION,
                    reason='CMOE_BENCHTRAIN_INTEGRATION=1 で走る')
def test_draws_a_stratified_set():
    """実際に引く。内訳が metadata に残り、同じ seed で同じ列が出る。"""
    from cmoe.data.benchtrain import calibration

    model = 'meta-llama/Llama-2-7b-hf'
    tokens = calibration(model, 2048, 8, 0)
    assert tuple(tokens.input_ids.shape) == (8, 2048)

    meta = tokens.metadata()
    assert meta['starts'] == []
    counts = {row['name']: row['n_sequences'] for row in meta['components']}
    assert counts == allocate(8)

    # プロセスをまたいで同じ列が出ることの担保。ここが変わったら、タスクの
    # 中身か引き方のどちらかが変わっている
    assert meta['token_hash'].startswith('7a0d5a5adf9e')

    other = calibration(model, 2048, 8, 1)
    assert other.metadata()['token_hash'] != meta['token_hash']


@pytest.mark.skipif(not INTEGRATION,
                    reason='CMOE_BENCHTRAIN_INTEGRATION=1 で走る')
def test_a_single_task_set_draws_only_that_task():
    """母集団が最も小さい ARC-Challenge だけで n=8 が引けることを実物で見る。"""
    from cmoe.data.registry import load_calibration

    tokens = load_calibration('benchtrain:arc_challenge',
                              'meta-llama/Llama-2-7b-hf', 2048, 8, 0)
    assert tuple(tokens.input_ids.shape) == (8, 2048)
    meta = tokens.metadata()
    assert [row['name'] for row in meta['components']] == ['arc_challenge']
    assert meta['components'][0]['n_sequences'] == 8
