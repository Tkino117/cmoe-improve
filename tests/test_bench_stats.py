"""選択問題の指標の確認（CPU、数値は手で出せるものだけ）。

このモジュールの存在理由は「正答率では見えない差を見る」ことなので、それが
実際に成り立つこと自体をテストにしてある
（``test_a_shift_too_small_to_move_accuracy_still_moves_gold_nll``）。
"""

import math

import pytest

from cmoe.eval import bench_stats
from cmoe.eval.bench import TaskSamples


def samples(loglikelihoods, gold, lengths=None, task='fake', hashes=None):
    n_docs = len(gold)
    return TaskSamples(
        task=task,
        gold=list(gold),
        loglikelihoods=[list(row) for row in loglikelihoods],
        choice_lengths=(lengths if lengths is not None
                        else [[1.0] * len(row) for row in loglikelihoods]),
        doc_hashes=(hashes if hashes is not None
                    else [f'doc{index}' for index in range(n_docs)]))


def test_log_softmax_is_a_distribution():
    values = bench_stats.log_softmax([-300.0, -301.5, -299.25])
    assert sum(math.exp(value) for value in values) == pytest.approx(1.0)


def test_gold_nll_is_the_negative_log_of_the_normalised_probability():
    # 選択肢の上での確率が 0.25 / 0.75 になるように置く
    one = samples([[math.log(0.25), math.log(0.75)]], gold=[1])
    assert bench_stats.gold_nll(one)[0] == pytest.approx(-math.log(0.75))
    other = samples([[math.log(0.25), math.log(0.75)]], gold=[0])
    assert bench_stats.gold_nll(other)[0] == pytest.approx(-math.log(0.25))


def test_the_sign_of_the_margin_is_the_accuracy():
    one = samples([[-1.0, -2.0], [-2.0, -1.0]], gold=[0, 0])
    assert bench_stats.accuracy(one) == [1.0, 0.0]
    margins = bench_stats.margin(one)
    assert margins[0] > 0 and margins[1] < 0


def test_length_normalisation_can_flip_the_answer():
    """acc と acc_norm は同じ尤度から出る別の量である。"""
    # 生では短い選択肢1が勝つ。長い正解肢は文字数で割ると逆転する
    # （対数尤度は負なので、長いほど割り算で 0 に近づく）
    one = samples([[-4.0, -3.0]], gold=[0], lengths=[[4.0, 1.0]])
    assert bench_stats.accuracy(one) == [0.0]
    assert bench_stats.accuracy_norm(one) == [1.0]


def test_a_shift_too_small_to_move_accuracy_still_moves_gold_nll():
    """このモジュールがある理由。

    全問の正解肢の尤度をわずかに下げる。マージンの符号は1問も変わらないので
    正答率は動かないが、K択NLL は問題ごとにきちんと動く。
    """
    baseline = samples([[-1.0, -3.0], [-2.0, -5.0], [-4.0, -4.5]], gold=[0, 0, 0])
    degraded = samples([[-1.02, -3.0], [-2.02, -5.0], [-4.02, -4.5]], gold=[0, 0, 0])

    assert bench_stats.accuracy(baseline) == bench_stats.accuracy(degraded)
    differences = bench_stats.paired_differences(baseline, degraded, 'acc')
    assert differences == [0.0, 0.0, 0.0]

    moved = bench_stats.paired_differences(baseline, degraded, 'gold_nll')
    assert all(value > 0 for value in moved)  # gold_nll は小さいほど良い


def test_the_kl_against_itself_is_zero():
    one = samples([[-1.0, -3.0, -2.0], [-2.0, -5.0, -1.0]], gold=[0, 2])
    assert bench_stats.reference_kl(one, one) == pytest.approx([0.0, 0.0])
    assert bench_stats.reference_agreement(one, one) == [1.0, 1.0]


def test_the_kl_is_positive_and_the_agreement_falls_when_the_choice_changes():
    reference = samples([[-1.0, -3.0]], gold=[0])
    candidate = samples([[-3.0, -1.0]], gold=[0])
    assert bench_stats.reference_kl(candidate, reference)[0] > 0
    assert bench_stats.reference_agreement(candidate, reference) == [0.0]


def test_a_different_question_order_is_refused():
    """対応のある比較の前提が崩れていたら、黙って平均せず止まる。"""
    one = samples([[-1.0, -2.0]], gold=[0], hashes=['a'])
    other = samples([[-1.0, -2.0]], gold=[0], hashes=['b'])
    with pytest.raises(ValueError, match='問題の並び'):
        bench_stats.reference_kl(one, other)


def test_the_macro_average_weights_tasks_equally():
    """問題数が一桁違うタスクを、問題数で重み付けしない。"""
    big = samples([[-1.0, -2.0]] * 10, gold=[0] * 10, task='big')     # 全問正解
    small = samples([[-2.0, -1.0]] * 2, gold=[0] * 2, task='small')   # 全問不正解
    rows = bench_stats.summarize({'big': big, 'small': small})
    assert rows['tasks']['big']['acc'] == 1.0
    assert rows['tasks']['small']['acc'] == 0.0
    # 問題数で重み付けするなら 10/12 = 0.833 になるところ
    assert rows['macro']['acc'] == pytest.approx(0.5)
    assert rows['n_docs'] == {'big': 10, 'small': 2}


def test_the_reference_metrics_are_dropped_without_a_reference():
    one = samples([[-1.0, -2.0]], gold=[0])
    assert set(bench_stats.per_doc(one)) == {'acc', 'acc_norm', 'gold_nll', 'margin'}
    assert set(bench_stats.per_doc(one, one)) == {
        'acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement'}


def test_the_difference_is_candidate_minus_baseline():
    baseline = samples([[-1.0, -2.0]], gold=[0])
    candidate = samples([[-1.5, -2.0]], gold=[0])
    # candidate の方が正解肢の確率が低い -> gold_nll は増える
    assert bench_stats.paired_differences(baseline, candidate, 'gold_nll')[0] > 0
    assert bench_stats.paired_differences(candidate, baseline, 'gold_nll')[0] < 0


@pytest.mark.parametrize('metric', sorted(bench_stats.HIGHER_IS_BETTER))
def test_every_metric_declares_its_direction(metric):
    """向きの表と、実際に出る指標が食い違わないこと。"""
    one = samples([[-1.0, -2.0]], gold=[0])
    assert metric in bench_stats.per_doc(one, one)
