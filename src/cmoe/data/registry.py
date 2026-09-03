"""データセット名の解決。

キャリブレーションと評価は別の表に分けてある。同じ名前で両方引ける必要はなく、
評価にしか使わないもの（c4-new）、carve にしか使わないものが今後も出るため。
"""

from cmoe.data import (benchchoice, benchqa, benchtrain, c4, flanv2,
                       slimpajama, wikitext2)

def _benchtrain_subset(label, tasks):
    """一部のタスクだけで引く benchtrain。本数は変えない（n_samples 本を配る）。

    表の行になるのはこちらで、``benchtrain``（5タスク混合）はその対照である。
    タスクが複数のときの配り方は ``benchtrain.allocate``（各タスクに1本、
    余りは並びの順）で、5タスクのときと同じ規則である。
    """
    def loader(model, seqlen, n_samples, seed):
        return benchtrain.calibration(model, seqlen, n_samples, seed,
                                      name=f'benchtrain-{label}', tasks=tasks)
    return loader


def _benchqa_subset(label, tasks):
    """一部のタスクだけで引く benchqa。採点位置の総量は変えない。"""
    def loader(model, seqlen, n_samples, seed):
        return benchqa.calibration(model, seqlen, n_samples, seed,
                                   name=f'benchqa-{label}', tasks=tasks)
    return loader


def _benchchoice_subset(label, tasks, budget_unit=None):
    """一部のタスクだけで引く benchchoice。予算の総量は変えない。"""
    def loader(model, seqlen, n_samples, seed):
        options = {} if budget_unit is None else {'budget_unit': budget_unit}
        return benchchoice.calibration(model, seqlen, n_samples, seed,
                                       name=f'benchchoice-{label}', tasks=tasks,
                                       **options)
    return loader


CALIBRATION_SETS = {
    'wikitext2': wikitext2.calibration,
    'c4': c4.calibration,
    'slimpajama': slimpajama.calibration,
    # 選択問題の train split。評価に使う split は入らない
    'benchtrain': benchtrain.calibration,
    # 同じものを1タスクに絞ったもの。``benchtrain:piqa`` のように書く
    **{f'benchtrain:{task}': _benchtrain_subset(task, (task,))
       for task in benchtrain.TASKS},
    # ARC は難易度で割った同じデータセットで、5×5 では互いに交換可能だった
    # （report/12）。ExpertWeaver Table 8 と同じく1タスクとして扱う行
    'benchtrain:arc': _benchtrain_subset('arc', ('arc_easy', 'arc_challenge')),
    # ExpertWeaver の多タスク校正（Flan、36タスク）を同じ量で写したもの
    'flanv2': flanv2.calibration,
    # 同じ train split を1問1系列で引き、採点に効く位置に印を付けたもの。
    # 印を使うかどうかは読む側（``--profile-positions``）が決める
    'benchqa': benchqa.calibration,
    **{f'benchqa:{task}': _benchqa_subset(task, (task,))
       for task in benchtrain.TASKS},
    'benchqa:arc': _benchqa_subset('arc', ('arc_easy', 'arc_challenge')),
    # 同じ問題を**選択肢ごとに1系列**で引いたもの。選択肢どうしを比べる目的
    # 関数 — ``--oracle margin`` — が要る唯一のセットである。予算はタスクごとの
    # 問題数で数える（マージンは問題ごとの量なので、位置で配ると問題数が続きの
    # 長さで決まってしまう）
    'benchchoice': benchchoice.calibration,
    **{f'benchchoice:{task}': _benchchoice_subset(task, (task,))
       for task in benchtrain.TASKS},
    'benchchoice:arc': _benchchoice_subset('arc', ('arc_easy', 'arc_challenge')),
    # 同じものを benchqa と同じ数え方（正解肢の採点位置）で引いたもの。同じ引数
    # なら benchqa とまったく同じ問題が選ばれるので、report/14 と校正の問題集合
    # を揃えたまま目的関数だけを替える対照に使う
    'benchchoice-qa': _benchchoice_subset('qa', benchtrain.TASKS,
                                          budget_unit='gold_scored'),
    # MMLU は57科目が lm-eval では別タスクで、``harness.subtasks`` が束ねる。
    # train split が無いので校正は dev + validation から引く（採点は test）。
    # ``benchqa`` 版は無い — MMLU の続きは選択肢の記号1文字なので、1問の採点
    # 位置が1つしかなく、母集団 1,816 問では既定の 16,384 位置に届かない
    'benchtrain:mmlu': _benchtrain_subset('mmlu', ('mmlu',)),
}

# ルーター方式が要る carve / fit / validation の3本組。
#
# slimpajama の carve は ``slimpajama.calibration`` と1トークンも変わらない
# （種に札を足さない経路を通す）。fit を要求する方式を足しても、分割と expert
# 重みは calibration だけで測った run と同一のままである。
SPLIT_SETS = {
    'wikitext2': wikitext2.splits,
    'slimpajama': slimpajama.splits,
}

EVALUATION_SETS = {
    'wikitext2': wikitext2.evaluation,
    'c4-new': c4.evaluation,
}


def load_calibration(name, model, seqlen, n_samples, seed):
    try:
        loader = CALIBRATION_SETS[name]
    except KeyError:
        raise ValueError(
            f'未知のキャリブレーションセット {name!r}。'
            f'{sorted(CALIBRATION_SETS)} から選ぶ') from None
    return loader(model, seqlen, n_samples, seed)


def load_evaluation(name, model, seqlen):
    try:
        loader = EVALUATION_SETS[name]
    except KeyError:
        raise ValueError(
            f'未知の評価セット {name!r}。{sorted(EVALUATION_SETS)} から選ぶ') from None
    return loader(model, seqlen)


def load_splits(name, model, seqlen, seed, carve_count=8, fit_count=64,
                validation_count=64):
    """carve / fit / validation の3本。ルーター方式を作るのに要る。"""
    try:
        loader = SPLIT_SETS[name]
    except KeyError:
        raise ValueError(
            f'{name!r} には carve/fit/validation の分割が無い。'
            f'{sorted(SPLIT_SETS)} から選ぶ') from None
    return loader(model, seqlen, seed, carve_count, fit_count, validation_count)
