"""[軸5] SA配分の境界。探索とスコアの2枚に割る。

配分 x は「その層で常時使う shared expert の数」であり、routed の Top-K は
``A - x`` になる。**x をどう決めても1トークンあたりに走る expert 数 A は
変わらない** — この不変が壊れると、配分ではなくモデルの大きさを比べたことに
なるので、ここで検査する。

境界が2枚に分かれているのが要点である。

* ``ScoreOracle``     … 配分候補 → (スコア, 消費コスト)。安い層ローカル指標から
                        実 PPL まで段階がある
* ``AllocationSearch`` … オラクルと予算 → 配分ベクトル。オラクルの中身は知らない

探索を足すのとオラクルを足すのが独立なので、「計算量を落とす探索」の比較が
同じ土俵でできる。
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Allocation:
    """層ごとの shared 数。層0 から順に並ぶ。"""

    values: tuple
    name: str = 'custom'
    n_active_total: int = 6

    def __post_init__(self):
        for index, x in enumerate(self.values):
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError(f'層 {index}: x は整数のはず (受け取った値: {x!r})')
            if not 0 <= x <= self.n_active_total:
                raise ValueError(
                    f'層 {index}: x={x} は 0..{self.n_active_total} の外。'
                    f'どの層も A={self.n_active_total} 個の expert を走らせる')

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def topk(self, index):
        """層 index の routed Top-K。A を固定したまま x と釣り合う値。"""
        return self.n_active_total - self.values[index]

    @property
    def mean_x(self):
        return sum(self.values) / len(self.values)

    def check_layers(self, n_layers):
        if len(self.values) != n_layers:
            raise ValueError(
                f'配分は {len(self.values)} 層分、モデルは {n_layers} 層')
        return self

    def metadata(self):
        return {
            'name': self.name,
            'values': list(self.values),
            'n_active_total': self.n_active_total,
            'mean_x': self.mean_x,
        }


@dataclass
class ScoreResult:
    """オラクル1回分の答え。

    score: 小さいほど良い量に揃える（PPL・KL・出力誤差はすべてそう）。
    cost:  そのスコアを得るのに使った量。探索どうしを「同じ予算で何が取れたか」
           で比べるための単位で、意味は ``cost_unit`` が持つ。
    """

    score: float
    cost: float = 0.0
    cost_unit: str = 'oracle_calls'
    details: dict = field(default_factory=dict)


class ScoreOracle(Protocol):
    """配分候補を採点する。中身の高価さは呼ぶ側から見えない。"""

    name: str
    cost_unit: str

    def score(self, allocation: Allocation) -> ScoreResult:
        ...

    @property
    def spent(self) -> float:
        """これまでに使った総コスト。"""


class AllocationSearch(Protocol):
    """配分を決める。オラクルの中身は知らない。"""

    name: str

    def search(self, oracle: ScoreOracle, n_layers: int) -> Allocation:
        ...
