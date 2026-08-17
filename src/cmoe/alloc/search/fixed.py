"""探索しない配分。オラクルを一度も叩かない。

* ``UniformSearch`` … 全層同じ x。ルーターだけを試したいときの既定。
* ``FixedSearch``   … 与えられたベクトルをそのまま返す。プリセットや、過去の
                      探索結果を再適用するときに使う。

探索を「何もしない」で差し替えられることが、ルーター単独の実験と配分込みの実験を
同じ組み立て役で表せる理由である。
"""

import re

from cmoe.alloc.base import Allocation
from cmoe.alloc.presets import N_ACTIVE, PRESETS


class UniformSearch:
    name = 'uniform'

    def __init__(self, x, n_active_total=N_ACTIVE):
        self.x = x
        self.n_active_total = n_active_total

    def search(self, oracle, n_layers):
        return Allocation(tuple([self.x] * n_layers), name=f'uniform{self.x}',
                          n_active_total=self.n_active_total)


class FixedSearch:
    name = 'fixed'

    def __init__(self, values, name='custom', n_active_total=N_ACTIVE):
        self.values = tuple(int(value) for value in values)
        self.allocation_name = name
        self.n_active_total = n_active_total

    @classmethod
    def from_preset(cls, name, n_active_total=N_ACTIVE):
        try:
            values = PRESETS[name]
        except KeyError:
            raise ValueError(
                f'未知の配分プリセット {name!r}。{sorted(PRESETS)} から選ぶ') from None
        return cls(values, name=name, n_active_total=n_active_total)

    def search(self, oracle, n_layers):
        return Allocation(self.values, name=self.allocation_name,
                          n_active_total=self.n_active_total).check_layers(n_layers)


UNIFORM = re.compile(r'^uniform(\d+)$')


def parse_allocation(spec, n_active_total=N_ACTIVE):
    """--alloc を解決する。

    受け付けるのは3種類 — ``uniform<x>``、プリセット名、カンマ区切りの値の並び。
    ``uniform<x>`` だけは層数に依存しないので、その場で組む。
    """
    uniform = UNIFORM.match(spec)
    if uniform:
        return UniformSearch(int(uniform.group(1)), n_active_total)
    if spec in PRESETS:
        return FixedSearch.from_preset(spec, n_active_total)
    fields = [field.strip() for field in spec.split(',') if field.strip()]
    if len(fields) < 2:
        raise ValueError(
            f'未知の配分 {spec!r}。uniform<x> か {sorted(PRESETS)} か、'
            '層数分のカンマ区切りの値')
    try:
        values = [int(field) for field in fields]
    except ValueError:
        raise ValueError(f'配分 {spec!r} が整数の並びになっていない') from None
    return FixedSearch(values, name='custom', n_active_total=n_active_total)
