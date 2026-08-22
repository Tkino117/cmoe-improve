"""選択問題の生の尤度から、指標を作る。GPU は要らない。

``bench`` が残すのは問題ごと・選択肢ごとの対数尤度だけで、指標はここが作る。
測り直さずに指標を足せるのが、生の値を残しておく理由である。

指標は6つ。上の2つが論文の表に載る量で、下の4つが「正答率では見えない差」を
見るためのものである。

======================  =====================================================
``acc``                 argmax が正解と一致したか（0/1）。論文の表の量
``acc_norm``            選択肢の文字数で割ってから argmax を取ったもの（0/1）
``gold_nll``            選択肢だけで softmax した正解確率の −log。K択分類の
                        交差エントロピー。連続量なので正答率より遥かに細かい
``margin``              正解の対数尤度 − 最良の不正解の対数尤度。符号が
                        ``acc`` そのもので、大きさが「どれだけ際どいか」
``ref_kl``              基準モデル（ふつう dense）との、選択肢上の分布の KL。
                        正誤と無関係に「どれだけ壊したか」を測る
``ref_agreement``       基準モデルと同じ選択肢を選んだか（0/1）。正答率と違い
                        正解・不正解どちらの向きの変化も拾う
======================  =====================================================

どれも**問題ごとの数**として返す。構成間で同じ問題が並ぶので、そのまま
``stats.stratified_paired_bootstrap`` の再抽出単位になる。
"""

import math

# 指標名 -> 大きいほど良いか。表示と、差の向きの読み方をここで一元化する
HIGHER_IS_BETTER = {
    'acc': True,
    'acc_norm': True,
    'gold_nll': False,
    'margin': True,
    'ref_kl': False,
    'ref_agreement': True,
}

# 基準モデルの測定が要る指標
NEEDS_REFERENCE = ('ref_kl', 'ref_agreement')


def log_softmax(values):
    """選択肢の対数尤度を、選択肢の上の対数確率に均す。

    max を引いてから exp するのは、対数尤度が −数百になるためである。
    """
    top = max(values)
    total = top + math.log(sum(math.exp(value - top) for value in values))
    return [value - total for value in values]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def accuracy(samples):
    return [1.0 if argmax(lls) == gold else 0.0
            for lls, gold in zip(samples.loglikelihoods, samples.gold)]


def accuracy_norm(samples):
    values = []
    for lls, lengths, gold in zip(samples.loglikelihoods, samples.choice_lengths,
                                  samples.gold):
        normalized = [ll / length for ll, length in zip(lls, lengths)]
        values.append(1.0 if argmax(normalized) == gold else 0.0)
    return values


def gold_nll(samples):
    """正解肢に置かれた確率の −log（選択肢の上で正規化したもの）。"""
    return [-log_softmax(lls)[gold]
            for lls, gold in zip(samples.loglikelihoods, samples.gold)]


def margin(samples):
    """正解と、最良の不正解との対数尤度の差。

    選択肢が1つしか無い問題は起こらない（multiple_choice の前提）。
    """
    values = []
    for lls, gold in zip(samples.loglikelihoods, samples.gold):
        others = [ll for index, ll in enumerate(lls) if index != gold]
        values.append(lls[gold] - max(others))
    return values


def check_aligned(samples, reference):
    """同じ問題が同じ順に並んでいるか。対応のある比較の前提そのもの。"""
    if samples.task != reference.task:
        raise ValueError(f'別のタスク: {samples.task} と {reference.task}')
    if samples.doc_hashes != reference.doc_hashes:
        raise ValueError(
            f'{samples.task}: 問題の並びが基準と違う（{samples.n_docs} 問 と '
            f'{reference.n_docs} 問）。対応のある比較にならない')


def reference_kl(samples, reference):
    """基準の選択肢分布から、こちらの分布への KL。

    向きは KL(基準 ‖ こちら)。配分探索の採点オラクル ``suffix_kl`` が
    dense を左に置いているのに合わせてある。
    """
    check_aligned(samples, reference)
    values = []
    for lls, base_lls in zip(samples.loglikelihoods, reference.loglikelihoods):
        log_p = log_softmax(base_lls)
        log_q = log_softmax(lls)
        values.append(sum(math.exp(left) * (left - right)
                          for left, right in zip(log_p, log_q)))
    return values


def reference_agreement(samples, reference):
    check_aligned(samples, reference)
    return [1.0 if argmax(lls) == argmax(base_lls) else 0.0
            for lls, base_lls in zip(samples.loglikelihoods, reference.loglikelihoods)]


def per_doc(samples, reference=None):
    """1タスク分の、指標名 -> 問題ごとの値。

    基準が無ければ基準の要る指標を落とす（dense を測っていない実行でも
    残りの4つは出る）。
    """
    values = {
        'acc': accuracy(samples),
        'acc_norm': accuracy_norm(samples),
        'gold_nll': gold_nll(samples),
        'margin': margin(samples),
    }
    if reference is not None:
        values['ref_kl'] = reference_kl(samples, reference)
        values['ref_agreement'] = reference_agreement(samples, reference)
    return values


def means(samples, reference=None):
    """1タスク分の指標の平均。"""
    return {name: sum(values) / len(values)
            for name, values in per_doc(samples, reference).items()}


def summarize(samples_by_task, reference_by_task=None):
    """タスクごとの平均と、タスク間の単純平均（マクロ平均）。

    マクロ平均を並べるのは、問題数がタスクで 1,200〜10,000 と一桁違い、
    まとめて平均すると HellaSwag の話になってしまうからである。
    """
    rows = {}
    for name, samples in samples_by_task.items():
        reference = reference_by_task.get(name) if reference_by_task else None
        rows[name] = means(samples, reference)
    metrics = sorted({name for row in rows.values() for name in row})
    macro = {name: sum(row[name] for row in rows.values() if name in row)
             / sum(1 for row in rows.values() if name in row)
             for name in metrics}
    return {'tasks': rows, 'macro': macro,
            'n_docs': {name: samples.n_docs
                       for name, samples in samples_by_task.items()}}


def paired_differences(baseline, candidate, metric, reference=None):
    """同じ問題どうしの、指標の差（1タスク分）。

    差の向きは常に「candidate − baseline」。良し悪しの向きは指標ごとに違う
    ので、読むときは ``HIGHER_IS_BETTER`` を見る。
    """
    check_aligned(candidate, baseline)
    left = per_doc(baseline, reference)[metric]
    right = per_doc(candidate, reference)[metric]
    return [after - before for before, after in zip(left, right)]
