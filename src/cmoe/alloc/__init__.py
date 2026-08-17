"""[軸5] SA配分の境界。"""

from cmoe.alloc.base import Allocation, AllocationSearch, ScoreOracle, ScoreResult
from cmoe.alloc.presets import PRESETS
from cmoe.alloc.search import FixedSearch, UniformSearch, parse_allocation

__all__ = ['Allocation', 'AllocationSearch', 'ScoreOracle', 'ScoreResult',
           'PRESETS', 'FixedSearch', 'UniformSearch', 'parse_allocation']
