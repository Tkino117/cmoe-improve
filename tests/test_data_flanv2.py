"""ExpertWeaver の多タスク校正（Flan）を写した校正セット。

CPU だけで見るのは3つ — タスクの並び（``TASKS``）、1タスクからの引き方
（``draw_task``）、タスクをまたぐ並べ方（``interleave``）である。実物を落として
引くところは ``CMOE_FLAN_INTEGRATION=1`` のときだけ走らせる。
"""

import os

import pytest

from cmoe.data.benchtrain import TASKS as BENCH_TASKS
from cmoe.data.flanv2 import TASKS, draw_task, interleave

INTEGRATION = os.environ.get('CMOE_FLAN_INTEGRATION') == '1'


def make_texts(count, size, tag='x'):
    return tuple(tag * size for _ in range(count))


# --- タスクの並び -----------------------------------------------------------

def test_the_task_list_has_no_duplicates():
    assert len(TASKS) == len(set(TASKS)) == 36


def test_four_of_the_five_benchmarks_are_in_the_calibration():
    """PIQA / HellaSwag / ARC-e / ARC-c は入り、WinoGrande だけ入らない。

    ミラーに WinoGrande が無いための欠けで、結果の読みに効くのでここで固定して
    おく（黙って入る／黙って消えるのを防ぐ）。
    """
    assert {'piqa', 'hellaswag', 'arc_easy', 'arc_challenge'} <= set(TASKS)
    assert 'winogrande' in BENCH_TASKS and 'winogrande' not in TASKS


# --- 1タスクからの引き方 ----------------------------------------------------

def test_it_takes_only_as_many_documents_as_the_quota_needs():
    texts = make_texts(400, 100)
    picked, size = draw_task(texts, 1000, 0, 'piqa')
    assert size >= 1000
    assert len(picked) == len(set(picked)) <= 11


def test_the_same_seed_draws_the_same_documents():
    texts = make_texts(400, 100)
    assert draw_task(texts, 1000, 0, 'piqa') == draw_task(texts, 1000, 0, 'piqa')
    assert draw_task(texts, 1000, 1, 'piqa') != draw_task(texts, 1000, 0, 'piqa')


def test_tasks_with_different_names_draw_differently():
    texts = make_texts(400, 100)
    assert draw_task(texts, 1000, 0, 'piqa') != draw_task(texts, 1000, 0, 'drop')


def test_a_pool_too_small_is_refused_instead_of_packing_tighter():
    with pytest.raises(ValueError, match='届かない'):
        draw_task(make_texts(4, 100), 1000, 0, 'wsc')


# --- タスクをまたぐ並べ方 ---------------------------------------------------

def test_documents_are_interleaved_across_tasks():
    """1周目に全タスクが1本ずつ出る。窓が1タスクに偏らないための性質。"""
    picked = {'a': (0, 1, 2), 'b': (10, 11), 'c': (20,)}
    order = interleave(picked, ('a', 'b', 'c'))
    assert order[:3] == [('a', 0), ('b', 10), ('c', 20)]
    assert order[3:] == [('a', 1), ('b', 11), ('a', 2)]
    assert len(order) == 6


# --- 実物 -------------------------------------------------------------------

@pytest.mark.skipif(not INTEGRATION, reason='CMOE_FLAN_INTEGRATION=1 で走る')
def test_draws_a_multi_task_set():
    """実際に落として引く。36タスクぶんの内訳が残り、seed で列が変わる。"""
    from cmoe.data.registry import load_calibration

    model = 'meta-llama/Llama-2-7b-hf'
    tokens = load_calibration('flanv2', model, 2048, 8, 0)
    assert tuple(tokens.input_ids.shape) == (8, 2048)

    meta = tokens.metadata()
    assert [row['name'] for row in meta['components']] == list(TASKS)
    assert len(meta['starts']) == 8

    other = load_calibration('flanv2', model, 2048, 8, 1)
    assert other.metadata()['token_hash'] != meta['token_hash']
