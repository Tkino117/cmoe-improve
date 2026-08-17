"""探索アルゴリズム名の解決。

``uniform<x>`` とプリセット、明示のベクトルはオラクルを叩かないので、ここには
入らない（``search/fixed.py`` の ``parse_allocation`` が扱う）。ここに並ぶのは
オラクルを叩いて配分を決める探索だけである。

幅の既定は探索ごとに違う（beam は4、greedy は1しか取れない）ので、``width`` は
既定を持たず、None なら各ファクトリが自分の既定を入れる。CLI が一律の既定を
持つと、``--search greedy`` が素で通らなくなる。
"""

from cmoe.alloc.presets import N_ACTIVE
from cmoe.alloc.search.beam import BeamSearch, GreedySearch

DEFAULT_BEAM_WIDTH = 4


def _beam(width=None, **kwargs):
    return BeamSearch(width=DEFAULT_BEAM_WIDTH if width is None else width,
                      **kwargs)


def _greedy(width=None, **kwargs):
    if width is not None and width != 1:
        # 幅を黙って無視すると、幅4 を指定したつもりの表が幅1 の結果になる
        raise ValueError(
            f'greedy は幅1のビームそのもの。--width {width} と両立しない '
            '（幅を変えたいなら --search beam）')
    return GreedySearch(**kwargs)


SEARCHES = {
    'beam': _beam,
    # 幅1のビームそのもの。別実装ではない
    'greedy': _greedy,
}


def create_search(name, width=None, n_active_total=N_ACTIVE, budget=None,
                  log=None, on_layer=None):
    check_search(name)
    return SEARCHES[name](width=width, n_active_total=n_active_total,
                          budget=budget, log=log, on_layer=on_layer)


def check_search(name, width=None):
    """名前（と、渡されていれば幅）だけを先に検査する。

    モデルを読み込む前に呼ぶためにある。7B を読み終えてから引数の綴り違いで
    落ちると、十数分が引数エラーのために消える。
    """
    if name not in SEARCHES:
        raise ValueError(f'未知の探索 {name!r}。{sorted(SEARCHES)} から選ぶ')
    if name == 'greedy' and width is not None and width != 1:
        raise ValueError(
            f'greedy は幅1のビームそのもの。--width {width} と両立しない '
            '（幅を変えたいなら --search beam）')
