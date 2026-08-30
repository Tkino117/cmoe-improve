"""MMLU を1タスクとして扱うための2つの規則。

MMLU は lm-eval では57科目が別々のタスクで、しかも train split が無い。
そこで足したのは2つだけである — 名前を科目に開く ``harness.subtasks`` と、
train split が無いタスクが校正に引いてよい split を決める
``harness.eligible_docs`` である。

科目の一覧はタスク定義（yaml）だけで決まるのでデータセットの取得は要らない。
実物を引くところは ``CMOE_BENCHTRAIN_INTEGRATION=1`` のときだけ走らせる。
"""

import os
from types import SimpleNamespace

import pytest

from cmoe.data.harness import (CALIBRATION_SPLITS, calibration_split,
                               eligible_docs, subtasks)

INTEGRATION = os.environ.get('CMOE_BENCHTRAIN_INTEGRATION') == '1'


def fake_task(name, splits, train=(), scored='test'):
    """``eligible_docs`` が読むところだけを持つタスク。"""
    return SimpleNamespace(
        dataset=splits,
        config=SimpleNamespace(task=name, test_split=scored,
                               validation_split=None),
        has_training_docs=lambda: bool(train),
        training_docs=lambda: list(train))


def test_mmlu_opens_into_57_subjects():
    found = subtasks('mmlu')
    assert len(found) == 57
    assert all(name.startswith('mmlu_') for name in found)
    # 並びが名前順に固定されていること。束ねたあとの問題の並びがこれで決まる
    assert list(found) == sorted(found)


def test_a_plain_task_opens_into_itself():
    assert subtasks('piqa') == ('piqa',)


def test_unknown_name_is_refused():
    with pytest.raises(ValueError):
        subtasks('mmlu_no_such_subject')


def test_train_split_is_the_default():
    task = fake_task('piqa', {'train': [1, 2]}, train=[1, 2])
    assert eligible_docs(task, None) == [1, 2]
    assert calibration_split('piqa') == 'train'


def test_named_splits_are_used_when_there_is_no_train():
    task = fake_task('mmlu_anatomy',
                     {'dev': ['d'], 'validation': ['v'], 'test': ['t']})
    assert eligible_docs(task, ('dev', 'validation')) == ['d', 'v']
    assert calibration_split('mmlu') == 'dev+validation'
    assert CALIBRATION_SPLITS['mmlu'] == ('dev', 'validation')


def test_the_scored_split_is_never_drawn():
    """採点する split を校正に引こうとしたら止まる。"""
    task = fake_task('mmlu_anatomy', {'dev': ['d'], 'test': ['t']})
    with pytest.raises(ValueError, match='採点に使う split'):
        eligible_docs(task, ('dev', 'test'))


def test_a_task_without_train_and_without_a_rule_is_refused():
    task = fake_task('mmlu_anatomy', {'dev': ['d'], 'test': ['t']})
    with pytest.raises(ValueError, match='train split が無く'):
        eligible_docs(task, None)


@pytest.mark.skipif(not INTEGRATION,
                    reason='CMOE_BENCHTRAIN_INTEGRATION=1 で走る')
def test_the_real_calibration_draws_from_dev_and_validation():
    """57科目を実際に引き、採点する test が1問も入らないことを見る。"""
    from cmoe.data.harness import DEFAULT_CACHE, load_tasks
    from cmoe.data.registry import load_calibration

    tokens = load_calibration('benchtrain:mmlu', 'meta-llama/Llama-2-7b-hf',
                              2048, 8, 0)
    assert tuple(tokens.input_ids.shape) == (8, 2048)
    component = tokens.metadata()['components'][0]
    assert component['name'] == 'mmlu'
    assert component['split'] == 'dev+validation'

    task = load_tasks(['mmlu_anatomy'], cache_dir=DEFAULT_CACHE)['mmlu_anatomy']
    drawn = {doc['question'] for doc in eligible_docs(task, ('dev', 'validation'))}
    scored = {doc['question'] for doc in task.dataset['test']}
    assert not drawn & scored
