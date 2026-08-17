"""配分の探索アルゴリズム。

オラクルを叩かないもの（``uniform<x>``・プリセット・明示のベクトル）は
``fixed.py``、叩いて配分を決めるもの（beam / greedy）は ``beam.py`` にある。
名前の解決は ``registry.py``（叩く側だけ）と ``parse_allocation``（叩かない側）。
"""

from cmoe.alloc.search.beam import BeamSearch, GreedySearch
from cmoe.alloc.search.fixed import FixedSearch, UniformSearch, parse_allocation

__all__ = ['BeamSearch', 'GreedySearch', 'FixedSearch', 'UniformSearch',
           'parse_allocation']
