"""データセット名の解決。

キャリブレーションと評価は別の表に分けてある。同じ名前で両方引ける必要はなく、
評価にしか使わないもの（c4-new）、carve にしか使わないものが今後も出るため。
"""

from cmoe.data import c4, slimpajama, wikitext2

CALIBRATION_SETS = {
    'wikitext2': wikitext2.calibration,
    'c4': c4.calibration,
    'slimpajama': slimpajama.calibration,
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
