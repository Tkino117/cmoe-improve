"""順序対照: 配分の**多重集合を保ったまま層の並びだけを崩す**。

**提案手法ではない。対照である。** そして、この土俵で作れる対照のなかで最も
強い形をしている。

探索が出した配分 ``v`` と、その並べ替え ``σ(v)`` を比べると、次のものが全部
同一になる。

* 1トークンあたりに走る expert 数 A（もともと x に依らない）
* 平均 x（``v`` と ``σ(v)`` は同じ多重集合）
* x の度数分布そのもの（``x=6`` の層が何本あるか、``x=0`` が何本か）
* 校正データ・分割・ルーター・評価・dense の基準

違うのは**どの層にどの x が載るか**だけである。したがって、ここに差が出れば
「層別に配分を決めることに意味がある」が言え、出なければ言えない。EW-rule や
一様配分との比較では、平均 x が動いてしまうためこの分離ができない
（report/21 の EW-rule は平均 x=2.41、提案は 4.60 で、差の一部は動作点の差で
説明できてしまう）。

対照は2種類ある。

* ``reverse``  … 層順の反転。決定的で、seed を持たない。層の深さと配分の関係
  （浅い層ほど shared が多い、など）が効いているなら、これが最も大きく壊す
* ``shuffle``  … 無作為な並べ替え。反転1本では「たまたま反転が悪い並び」を
  引いた可能性が残るので、独立な引きを複数本置く

どちらも**恒等置換にならないことを確かめてから返す**。全層同じ x のベクトル
（一様配分）はどう並べ替えても同じなので、その場合は例外にする — 潰れた対照を
数時間ベンチに掛けて「差が無い」と読むのが、この実験で最も高くつく失敗である。
"""

import random


def reverse(values):
    """層順の反転。"""
    values = tuple(values)
    _reject_uniform(values, 'reverse')
    return tuple(reversed(values))


def shuffle(values, seed, max_tries=64):
    """無作為な並べ替え。``seed`` が同じなら同じ並びを返す。

    恒等置換と反転はどちらも別の対照として持っているので、引き直す。``values``
    に同じ値が多いと引き直しが続くことがあるため上限を置くが、上限に当たるのは
    実質「多重集合がほぼ一様」のときで、そのときは ``reverse`` も無意味である。
    """
    values = tuple(values)
    _reject_uniform(values, 'shuffle')
    rng = random.Random(seed)
    backward = tuple(reversed(values))
    for _ in range(max_tries):
        candidate = list(values)
        rng.shuffle(candidate)
        candidate = tuple(candidate)
        if candidate != values and candidate != backward:
            return candidate
    raise ValueError(
        f'{max_tries} 回引いても、元とも反転とも違う並びが出ない。'
        f'多重集合 {sorted(set(values))} が偏りすぎている')


def _reject_uniform(values, what):
    if len(set(values)) <= 1:
        raise ValueError(
            f'全層が x={values[0] if values else None} の配分は、'
            f'{what} しても同じベクトルになる。順序対照にならない')


def displacement(values, permuted):
    """並べ替えでどれだけ動いたか。表に添える数。

    ``n_changed`` は x が変わった層の数、``mean_abs_shift`` は層ごとの |Δx| の
    平均である。0 なら並べ替えが効いていない。
    """
    values, permuted = tuple(values), tuple(permuted)
    if len(values) != len(permuted):
        raise ValueError(f'長さが違う（{len(values)} と {len(permuted)}）')
    if sorted(values) != sorted(permuted):
        raise ValueError('多重集合が違う。これは並べ替えではない')
    shifts = [abs(a - b) for a, b in zip(values, permuted)]
    return {
        'n_changed': sum(1 for s in shifts if s),
        'mean_abs_shift': sum(shifts) / len(shifts) if shifts else 0.0,
        'max_abs_shift': max(shifts) if shifts else 0,
    }
