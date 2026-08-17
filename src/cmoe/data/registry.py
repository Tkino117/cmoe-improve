"""データセット名の解決。

キャリブレーションと評価は別の表に分けてある。同じ名前で両方引ける必要はなく、
評価にしか使わないもの（c4-new）、carve にしか使わないものが今後も出るため。
"""

from cmoe.data import c4, wikitext2

CALIBRATION_SETS = {
    'wikitext2': wikitext2.calibration,
    'c4': c4.calibration,
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
