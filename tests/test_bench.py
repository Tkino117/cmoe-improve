"""lm-eval からの取り出しの確認（CPU、ネットワーク不要）。

ここが守っているのは1点だけ — **正解番号の導出**である。``bench._gold_index``
は lm-eval の ``api/task.py`` の写しなので、写し間違いがあれば指標はすべて
意味を失う。本番では ``_check_against_harness`` が全問について lm-eval 自身の
正誤と突き合わせるので、そのしくみが働くこと（正しければ通り、ずらせば止まる）
をここで見る。

実物のタスクは使わない。データセットの取得が要るうえ、確かめたいのは
「lm-eval が残したログから正解番号を復元できるか」であって、PIQA の中身では
ないからである。
"""

import os
import random
import types

import numpy
import pytest
import torch

from cmoe.eval import bench


class FakeTask:
    """``doc_to_*`` だけを持つ、lm-eval のタスクの代わり。

    ``multiple_input`` は WinoGrande の形（選択肢が「続き」ではなく「文脈」の
    側に立ち、正解番号が ``doc_to_text`` から来る）を表す。
    """

    def __init__(self, name, target_kind, multiple_input=0):
        self.config = types.SimpleNamespace(task=name)
        self.multiple_input = multiple_input
        self._target_kind = target_kind

    def doc_to_choice(self, doc):
        return doc['choices']

    def doc_to_text(self, doc):
        return doc['gold']

    def doc_to_target(self, doc):
        if self._target_kind == 'int':      # ARC / PIQA / HellaSwag
            return doc['gold']
        if self._target_kind == 'str':      # 選択肢の文字列そのもの
            return doc['choices'][doc['gold']]
        raise AssertionError(self._target_kind)


def example(doc_id, choices, gold, lls, acc=None, acc_norm=None):
    """lm-eval が ``log_samples`` で残すのと同じ形の1問分。"""
    lengths = [float(len(choice)) for choice in choices]
    if acc is None:
        acc = 1.0 if max(range(len(lls)), key=lls.__getitem__) == gold else 0.0
    if acc_norm is None:
        normalized = [ll / length for ll, length in zip(lls, lengths)]
        acc_norm = 1.0 if max(range(len(normalized)),
                              key=normalized.__getitem__) == gold else 0.0
    return {
        'doc_id': doc_id,
        'doc': {'choices': choices, 'gold': gold},
        'filtered_resps': [(ll, False) for ll in lls],
        'doc_hash': f'hash{doc_id}',
        'acc': acc,
        'acc_norm': acc_norm,
    }


DOCS = [
    (['alpha', 'beta longer'], 0, [-1.0, -3.0]),
    (['gamma', 'delta'], 1, [-4.0, -2.0]),
    (['one', 'two', 'three', 'four'], 2, [-5.0, -6.0, -1.5, -9.0]),
]


@pytest.mark.parametrize('target_kind', ['int', 'str'])
def test_the_gold_index_is_recovered_however_the_task_states_it(target_kind):
    task = FakeTask('fake', target_kind)
    examples = [example(index, choices, gold, lls)
                for index, (choices, gold, lls) in enumerate(DOCS)]
    gold, lls, lengths, hashes = bench._extract('fake', task, examples)
    assert gold == [row[1] for row in DOCS]
    assert lls == [row[2] for row in DOCS]
    assert lengths == [[float(len(choice)) for choice in row[0]] for row in DOCS]
    assert hashes == ['hash0', 'hash1', 'hash2']


def test_the_gold_index_of_a_multiple_input_task_comes_from_the_context_side():
    """WinoGrande の形。``doc_to_target`` は続きの文字列で、番号ではない。"""
    task = FakeTask('winogrande-like', 'str', multiple_input=2)
    doc = {'choices': ['ctx a', 'ctx b'], 'gold': 1}
    assert bench._gold_index(task, doc, 2) == 1


def test_a_wrong_gold_index_is_caught_against_the_harness():
    """写し間違いが黙って通らないこと。

    lm-eval 側の acc を意図的にひっくり返す。導出が食い違えば止まる — これが
    ``_gold_index`` を自前で持ってよい唯一の理由である。
    """
    task = FakeTask('fake', 'int')
    bad = example(0, ['alpha', 'beta'], 0, [-1.0, -3.0], acc=0.0, acc_norm=0.0)
    with pytest.raises(ValueError, match='食い違う'):
        bench._extract('fake', task, [bad])


def test_documents_are_ordered_by_doc_id():
    """並び順が構成間で揃うこと。対応のある比較の前提。"""
    task = FakeTask('fake', 'int')
    examples = [example(index, choices, gold, lls)
                for index, (choices, gold, lls) in enumerate(DOCS)]
    _, _, _, hashes = bench._extract('fake', task, list(reversed(examples)))
    assert hashes == ['hash0', 'hash1', 'hash2']


def test_a_gold_index_outside_the_choices_is_refused():
    task = FakeTask('fake', 'int')
    with pytest.raises(ValueError, match='収まらない'):
        bench._gold_index(task, {'choices': ['a', 'b'], 'gold': 5}, 2)


def test_the_task_samples_survive_a_round_trip():
    task = FakeTask('fake', 'int')
    examples = [example(index, choices, gold, lls)
                for index, (choices, gold, lls) in enumerate(DOCS)]
    gold, lls, lengths, hashes = bench._extract('fake', task, examples)
    one = bench.TaskSamples(task='fake', gold=gold, loglikelihoods=lls,
                            choice_lengths=lengths, doc_hashes=hashes,
                            harness_metrics={'acc,none': 0.5}, num_fewshot=0)
    other = bench.TaskSamples.from_dict(one.as_dict())
    assert other.as_dict() == one.as_dict()
    assert other.n_docs == 3


def test_the_global_random_state_is_left_where_it_was():
    """lm-eval は大域シードを踏む。測る道具が測る対象を動かさないこと。"""
    random.seed(1234)
    numpy.random.seed(1234)
    torch.manual_seed(1234)
    before = (random.random(), float(numpy.random.rand()), float(torch.rand(1)))

    random.seed(1234)
    numpy.random.seed(1234)
    torch.manual_seed(1234)
    with bench._isolated_rng():
        random.seed(99)
        numpy.random.seed(99)
        torch.manual_seed(99)
        random.random(), numpy.random.rand(), torch.rand(1)
    after = (random.random(), float(numpy.random.rand()), float(torch.rand(1)))
    assert after == before


# ---------------------------------------------------------------- 統合の確認

# 本物の lm-eval を通す確認。データセットの取得とタスク5つの構築で20秒ほど
# かかるので、既定の「CPU で数秒」の suite からは外してある。ベンチ経路
# （bench.py / bench_stats.py / CLI の --bench）に触ったら、これを通すこと:
#
#   CMOE_BENCH_INTEGRATION=1 uv run pytest tests/test_bench.py -q
INTEGRATION = os.environ.get('CMOE_BENCH_INTEGRATION') == '1'


@pytest.mark.skipif(not INTEGRATION, reason='CMOE_BENCH_INTEGRATION=1 で走る')
def test_the_real_harness_agrees_with_us_on_all_five_tasks():
    """極小のランダムモデルで、5タスクを本物の lm-eval に通す。

    数値そのものは見ない（ランダムな重みなので意味が無い）。見るのは2つ。

    * ``_check_against_harness`` が全問で通ること。つまり ``_gold_index`` の
      写経が、本物のタスク設定（ARC の int、PIQA の ClassLabel、WinoGrande の
      ``multiple_input``）すべてと一致していること
    * こちらが問題ごとの値から出した平均が、lm-eval 自身の集計値と一致すること
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from cmoe.data.base import load_tokenizer
    from cmoe.eval import bench_stats

    tokenizer = load_tokenizer('meta-llama/Llama-2-7b-hf')
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.float32)
    model.eval()
    model.config.use_cache = False

    samples = bench.evaluate_bench(
        model, tokenizer, batch_size=4, limit=4, max_length=2048,
        cache_dir='.cache/hf-datasets')

    assert set(samples) == set(bench.DEFAULT_TASKS)
    for name, row in samples.items():
        assert row.n_docs == 4, name
        mine = bench_stats.means(row)
        for metric in ('acc', 'acc_norm'):
            key = f'{metric},none'
            if key in row.harness_metrics:
                assert mine[metric] == pytest.approx(row.harness_metrics[key]), (
                    name, metric)

    # 自分自身を基準に置けば、壊した量はゼロで、選ぶ答えは全問一致する
    rows = bench_stats.summarize(samples, samples)
    assert rows['macro']['ref_kl'] == pytest.approx(0.0)
    assert rows['macro']['ref_agreement'] == 1.0


def test_saved_samples_can_be_read_back(tmp_path):
    """一次データだけで指標を作り直せること。

    レポートを書く経路も、指標を後から足す経路もここを通る。保存されたものが
    ``TaskSamples`` に戻らなければ、GPU で測り直す以外に手が無くなる。
    """
    import json

    from cmoe.eval import bench_stats

    task = FakeTask('fake', 'int')
    examples = [example(index, choices, gold, lls)
                for index, (choices, gold, lls) in enumerate(DOCS)]
    gold, lls, lengths, hashes = bench._extract('fake', task, examples)
    one = bench.TaskSamples(task='fake', gold=gold, loglikelihoods=lls,
                            choice_lengths=lengths, doc_hashes=hashes)

    path = tmp_path / 'bench.json'
    path.write_text(json.dumps({'fake': one.as_dict()}))
    loaded = bench.load_samples(path)

    assert set(loaded) == {'fake'}
    # 読み戻したものから出る指標が、元のものと1ビットも変わらないこと
    assert bench_stats.per_doc(loaded['fake'], loaded['fake']) == \
        bench_stats.per_doc(one, one)
