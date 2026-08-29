"""データセット名の解決。

キャリブレーションと評価は別の表に分けてある。同じ名前で両方引ける必要はなく、
評価にしか使わないもの（c4-new）、carve にしか使わないものが今後も出るため。
"""

from cmoe.data import benchqa, benchtrain, c4, flanv2, slimpajama, wikitext2

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
}

# ルーター方式が要る carve / fit / validation の3本組。
SPLIT_SETS = {
    'wikitext2': wikitext2.splits,
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
