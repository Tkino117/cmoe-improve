"""配分の探索。合成オラクルで、枝刈りの規則そのものを見る。

モデルもトークンも要らない。ビームが持つべき性質 — 貪欲が取り逃がす答えを拾う、
同点で揺れない、状態を捨て忘れない、予算を超えたら止まる — はすべて「スコアを
返す関数」だけで決まるからで、13GB のモデルを読まないと動かせない盤面は動作確認
されない盤面である。
"""

import pytest

from cmoe.alloc.base import Allocation, ScoreResult
from cmoe.alloc.search.beam import (BeamEntry, BudgetExceeded, GreedySearch,
                                    TopBeam, parent_ranks)
from cmoe.alloc.search.fixed import parse_allocation
from cmoe.alloc.search.registry import create_search

# 貪欲が必ず外す盤面。層0 は x=1 がその場の最善（0.5）だが、その先は 0.9 までしか
# 下がらない。x=0 は層0 で見劣りする（1.0）のに、層1 で 0.1 に届く。
TRAP = {
    (0,): 1.0, (1,): 0.5, (2,): 2.0,
    (0, 0): 0.1, (0, 1): 0.4, (0, 2): 0.7,
    (1, 0): 0.9, (1, 1): 1.2, (1, 2): 1.5,
    (2, 0): 1.8, (2, 1): 1.9, (2, 2): 2.5,
}


class FakeOracle:
    """接頭辞 → スコアの表を引くだけのオラクル。状態は接頭辞そのもの。"""

    name = 'fake'
    cost_unit = 'calls'

    def __init__(self, scores=None, n_candidates=3, n_active_total=2, cost=1.0):
        self.scores = TRAP if scores is None else scores
        self.n_candidates = n_candidates
        self.n_active_total = n_active_total
        self.cost = cost
        self.released = []
        self._spent = 0.0
        self._calls = 0

    @property
    def spent(self):
        return self._spent

    @property
    def calls(self):
        return self._calls

    def candidates(self, layer):
        return tuple(range(self.n_candidates))

    def root(self):
        return ()

    def release(self, state):
        if state is not None:
            self.released.append(state)

    def extend(self, state, layer, x):
        if len(state) != layer:
            raise AssertionError(f'深さ {len(state)} の接頭辞を層 {layer} で展開した')
        prefix = state + (x,)
        self._spent += self.cost
        self._calls += 1
        return (ScoreResult(score=self.scores[prefix], cost=self.cost,
                            cost_unit=self.cost_unit, details={'x': x}),
                prefix)


# -- 盤面 -------------------------------------------------------------------


def test_allocation_is_read_off_the_lineage():
    root = BeamEntry.root(payload='in')
    child = root.extend(3, 0.5)
    grandchild = child.extend(1, 0.2)
    assert root.allocation == ()
    assert grandchild.allocation == (3, 1)
    assert grandchild.depth == 2
    assert root.is_root and not child.is_root


def test_beam_keeps_the_best_and_evicts_the_rest():
    evicted = []
    beam = TopBeam(2, on_evict=evicted.append)
    root = BeamEntry.root()
    for x, score in ((0, 0.3), (1, 0.1), (2, 0.2)):
        beam.insert(root.extend(x, score))
    assert [entry.allocation for entry in beam.entries] == [(1,), (2,)]
    assert beam.best.score == 0.1
    # 落ちたのは1つだけで、落ちた瞬間に1度だけ通知される
    assert [entry.allocation for entry in evicted] == [(0,)]


def test_beam_reports_the_entry_that_lost_even_when_it_just_arrived():
    beam = TopBeam(1)
    root = BeamEntry.root()
    assert beam.insert(root.extend(0, 0.1)) is None
    loser = beam.insert(root.extend(1, 0.9))
    assert loser.allocation == (1,)


def test_ties_break_on_x_then_on_arrival():
    beam = TopBeam(3)
    root = BeamEntry.root()
    for x in (2, 0, 1):
        beam.insert(root.extend(x, 0.5))
    # 同点なら x の小さい順。親を展開する順序に左右されない同点崩し
    assert [entry.x for entry in beam.entries] == [0, 1, 2]

    other = TopBeam(2)
    # 親が違えば、同じ x・同じスコアの子が2つありうる（配分は別物になる）
    first = BeamEntry.root().extend(2, 0.4)
    second = BeamEntry.root().extend(0, 0.4)
    other.insert(first.extend(1, 0.5))
    other.insert(second.extend(1, 0.5))
    # x も同じなら到着順。sort の同点の扱いに依存しない
    assert [entry.allocation for entry in other.entries] == [(2, 1), (0, 1)]


def test_the_same_allocation_cannot_be_offered_twice():
    beam = TopBeam(4)
    root = BeamEntry.root()
    beam.insert(root.extend(1, 0.5))
    with pytest.raises(ValueError, match='既に差し出されている'):
        beam.insert(root.extend(1, 0.7))


def test_an_unscored_child_cannot_be_ranked():
    beam = TopBeam(2)
    with pytest.raises(ValueError, match='スコアが無い'):
        beam.insert(BeamEntry.root().extend(1, None))


def test_parent_ranks_matches_by_identity():
    first, second = BeamEntry.root(), BeamEntry.root()
    entries = [second.extend(1, 0.2), first.extend(1, 0.2)]
    assert parent_ranks(entries, [first, second]) == [(1, 1), (0, 1)]
    with pytest.raises(ValueError, match='別の層のもの'):
        parent_ranks(entries, [first])


# -- 歩き方 -----------------------------------------------------------------


def test_beam_finds_what_greedy_misses():
    greedy = create_search('greedy', width=1)
    assert greedy.search(FakeOracle(), 2).values == (1, 0)      # その場の最善を追う

    beam = create_search('beam', width=2)
    assert beam.search(FakeOracle(), 2).values == (0, 0)        # 最適
    # 同じ盤面・同じオラクルで、違うのは幅だけ
    assert isinstance(greedy, GreedySearch)


def test_search_returns_an_allocation_with_the_oracle_s_budget_of_experts():
    oracle = FakeOracle()
    allocation = create_search('beam', width=2).search(oracle, 2)
    assert isinstance(allocation, Allocation)
    assert len(allocation) == 2
    assert allocation.n_active_total == oracle.n_active_total
    assert allocation.name == 'beam2'
    assert allocation.topk(0) == oracle.n_active_total - allocation[0]


def test_every_state_is_handed_back_exactly_once():
    oracle = FakeOracle()
    allocation = create_search('beam', width=2).search(oracle, 2)
    # 根 + 層0 の子3本 + 層1 の子6本（生き残り2本を含む）
    assert len(oracle.released) == 1 + 3 + 6
    assert len(set(oracle.released)) == len(oracle.released)
    assert tuple(allocation.values) in oracle.released


def test_the_layer_record_says_what_the_pruning_decided():
    search = create_search('beam', width=2)
    search.search(FakeOracle(), 2)
    first, second = search.records
    assert first['layer'] == 0 and len(first['rows']) == 1
    assert [row['x'] for row in first['beam']] == [1, 0]
    # 落ちた中の最良（2.0）と残った中の最悪（1.0）の差
    assert first['margin'] == pytest.approx(1.0)
    assert first['best_dropped'] == pytest.approx(2.0)
    # 層1 は親2本 × 候補3個 = 6 本を測る
    assert sum(len(row['children']) for row in second['rows']) == 6
    assert second['spent'] == 9.0


def test_a_wider_beam_than_the_children_offered_drops_nothing():
    search = create_search('beam', width=8)
    search.search(FakeOracle(), 2)
    assert search.records[0]['margin'] is None


def test_the_layer_record_is_handed_over_as_each_layer_finishes():
    seen = []
    search = create_search('beam', width=2, on_layer=lambda records: seen.append(len(records)))
    search.search(FakeOracle(), 2)
    # 層が終わるたびに1回。落ちても終わった層は書き出されている
    assert seen == [1, 2]


def test_budget_stops_the_walk_and_keeps_what_was_decided():
    oracle = FakeOracle()
    search = create_search('beam', width=2, budget=3.0)
    with pytest.raises(BudgetExceeded) as error:
        search.search(oracle, 2)
    # 層0 で3回叩いて予算に届き、層1 に入る前に止まる
    assert error.value.spent == 3.0
    assert error.value.prefix == (1,)
    # 途中で止めても状態は手放す。握ったまま抜けると呼び出し側から解放できない
    assert set(oracle.released) == {(), (0,), (1,), (2,)}


def test_a_failing_oracle_does_not_strand_the_states():
    """層の途中で落ちても、まだ展開していない親と集めかけの子を捨てる。"""
    class Failing(FakeOracle):
        def extend(self, state, layer, x):
            if layer == 1 and x == 2:
                raise RuntimeError('測定に失敗した')
            return super().extend(state, layer, x)

    oracle = Failing()
    with pytest.raises(RuntimeError, match='測定に失敗した'):
        create_search('beam', width=2).search(oracle, 2)
    # 層0 の根と子3本、層1 で作りかけた子。どれも解放済みで、重複して返さない
    assert len(oracle.released) == len(set(oracle.released))
    assert () in oracle.released and (1,) in oracle.released


def test_greedy_runs_without_being_told_its_width():
    """``--search greedy`` が素で通る。

    幅の既定を CLI が一律に持つと（beam 用の 4）、greedy は必ず幅の不一致で
    落ちる。既定は探索ごとに解決する。
    """
    assert create_search('greedy').width == 1
    assert create_search('beam').width == 4


def test_greedy_refuses_a_width_it_would_have_ignored():
    with pytest.raises(ValueError, match='幅1のビーム'):
        create_search('greedy', width=4)


def test_unknown_search_names_are_refused():
    with pytest.raises(ValueError, match='未知の探索'):
        create_search('annealing')


def test_a_searched_allocation_can_be_pasted_into_run():
    """``search`` が印字したベクトルを ``run --alloc`` がそのまま受ける。

    ``--alloc`` のカンマは2つの意味を持つ — 配分の並び（直積に回す）と、層ごとの
    値を並べた1本のベクトル。区別が無いと、32層のベクトルが32個の未知の配分名に
    なり、探索の結果を使う経路がそこで切れる。実機で踏んだ。
    """
    from cmoe.cli import build_parser, configurations, parse_alloc_specs

    searched = '3,6,6,6,5,5,5,6,6,6,6,6,6,6,6,5,6,5,6,5,5,6,6,6,6,6,4,4,2,3,4,5'
    assert parse_alloc_specs(searched) == [searched]      # 1本のベクトル
    assert parse_alloc_specs('uniform3,beam') == ['uniform3', 'beam']
    assert parse_alloc_specs('beam') == ['beam']

    # 直積の側から見ても1構成にしかならない
    args = build_parser().parse_args(
        ['run', '--alloc', searched, '--router', 'cmoe,oracle_recovery'])
    assert configurations(args) == [(searched, 'cmoe'),
                                    (searched, 'oracle_recovery')]
    # そのベクトルが実際に配分として解ける
    allocation = parse_allocation(searched, n_active_total=6).search(None, 32)
    assert len(allocation) == 32
    assert allocation[0] == 3


def test_argument_errors_surface_before_the_model_is_loaded():
    """引数だけで分かる誤りは、13GB を読み込む前に出し切る。

    ここを通さないと、綴り違いで落ちるのがモデル読み込みとキャリブレーション
    捕捉のあと — 7B なら十数分後になる。
    """
    from cmoe.cli import build_parser, check_search_arguments

    def parse(*extra):
        return build_parser().parse_args(['search', *extra])

    check_search_arguments(parse('--search', 'greedy'))      # 素で通る
    with pytest.raises(ValueError, match='未知の採点オラクル'):
        check_search_arguments(parse('--oracle', 'l_table'))
    with pytest.raises(ValueError, match='未知の探索'):
        check_search_arguments(parse('--search', 'annealing'))
    with pytest.raises(ValueError, match='幅1のビーム'):
        check_search_arguments(parse('--search', 'greedy', '--width', '4'))
    # --layers 0 は falsy なので、検査が無いと「全層」に化ける
    with pytest.raises(SystemExit, match='--layers'):
        check_search_arguments(parse('--layers', '0'))
    with pytest.raises(SystemExit, match='何も測らない'):
        check_search_arguments(
            parse('--oracle', 'mass', '--nexperts', '4', '--nactive', '4'))


def test_a_search_without_an_oracle_is_refused():
    with pytest.raises(ValueError, match='オラクルを要求'):
        create_search('beam', width=2).search(None, 2)
