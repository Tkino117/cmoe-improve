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

オラクルの契約は2つある。``ScoreOracle`` は配分ベクトル1本を丸ごと採点する
（実 PPL のように、全層が決まって初めて意味を持つ指標がこれ）。実際に走る探索
— greedy も beam も — が要求するのはもう一方の ``PrefixOracle`` で、層0 から
順に「ここまでの接頭辞に x を足したら」を採点する。前から進める探索が接頭辞を
毎回ゼロから測り直さずに済むのは、この形だけである。

接頭辞の**状態**（そこまでの層が次の層へ渡す隠れ状態）は探索から見て不透明で
ある。探索が持つのは「どの状態から x を足したか」という系譜だけで、状態の中身
を読まない。これは節約のためではなく境界の位置の問題で、beam の枝刈り規則は
GPU も 13GB のモデルも無しに検査できるところに置く。
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


class PrefixOracle(Protocol):
    """層0 から順に、接頭辞へ1層足したものを採点する。

    ``extend`` が返すスコアは**その接頭辞全体**のスコアであって、足した1層の
    ぶんではない。層をまたいでどう積むか（層ローカル誤差なら足し合わせ、suffix
    KL なら測り直し）は指標ごとに違い、探索がそれを知る必要はないからである。
    同じ層で比べられる限り、探索は大小しか見ない。

    ``state`` は不透明な接頭辞の状態。作るのも捨てるのもオラクルで、探索は
    受け取った物をそのまま持ち回り、要らなくなったら ``release`` に返す。
    """

    name: str
    cost_unit: str

    def candidates(self, layer: int) -> tuple:
        """その層で試せる x。"""

    def root(self) -> object:
        """何も決めていない接頭辞の状態（= キャリブレーション入力）。"""

    def extend(self, state, layer: int, x: int) -> tuple:
        """(ScoreResult, 新しい状態)。スコアは小さいほど良い。"""

    def release(self, state) -> None:
        """この状態はもう読まれない、と伝える。"""

    @property
    def spent(self) -> float:
        """これまでに使った総コスト。"""


class AllocationSearch(Protocol):
    """配分を決める。オラクルの中身は知らない。"""

    name: str

    def search(self, oracle: PrefixOracle, n_layers: int) -> Allocation:
        ...
