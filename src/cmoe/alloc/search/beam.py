"""層を前から順に決める。良い接頭辞を B 本だけ生かして進むビーム探索。

CMoE-ref の ``xsearch/beam.py`` と ``scripts/beam_search.py`` の探索部分の移送。
貪欲な歩き方は各層でその場の最善を1つ取り、二度と振り返らない。順に効いていく
モデルでは、それがちょうど間違える場合がある — 層3 で最善に見えた選択が、
層4以降に何も残さない選択でありうるし、1本しか持たない歩き方はそれに気づけない。
ここでは B 本を残すので、2番手に見えた層が、後続の層に評価されるまで生き延びる。

**幅と深さは別の軸である。** 幅 B は「同じ層で何本残すか」、深さ（``lookahead``）は
「刈る前に何層先まで展開するか」である。既定の深さ0 では、層 ℓ の子はその場の
スコアだけで順位が付く — 層 ℓ の選択が層 ℓ+1 に何を残すかは、刈ったあとにしか
分からない。深さ1 では子ごとに次の層の候補を全部展開し、**その最良**を子の順位に
使う。``lookahead=0`` は既定のビームそのもので、同じコードが走る。

**B = 1 が貪欲な歩き方である。** 幅1のビームでは各層の生存者が親の候補の argmin
になり、それは貪欲そのものである。特別扱いではなく退化した場合で、同じコードが
走る。「ビームが貪欲では見つからないものを見つけた」が、2本のスクリプトの違い
ではなく探索の違いについての主張になるのはそのためである。

この盤面（``BeamEntry`` / ``TopBeam``）に torch は要らない。枝刈りの順序も、
接頭辞の系譜も、隠れ状態が要らなくなる瞬間も、GPU 抜きで決まる。13GB のモデルを
読み込まないと動かせない盤面は、動作確認されない盤面である。
"""

from dataclasses import dataclass, field

from cmoe.alloc.base import Allocation
from cmoe.alloc.presets import N_ACTIVE


class BudgetExceeded(Exception):
    """予算を使い切ったので探索を止めた。

    途中結果を捨てないよう、そこまでで最良の接頭辞を持たせる。探索どうしを
    「同じ予算で何が取れたか」で比べるとき、超過を黙って許すと比較が壊れる。
    """

    def __init__(self, message, prefix=(), spent=0.0, budget=0.0):
        super().__init__(message)
        self.prefix = tuple(prefix)
        self.spent = spent
        self.budget = budget


@dataclass
class BeamEntry:
    """配分の接頭辞1本。層 0..depth-1 が決まり、残りは未定。

    x:       **最後に決めた層**の shared 数。根は None（何も決めていない）。
    score:   その展開が測った接頭辞のスコア。根は None。
    payload: 不透明。オラクルが作った接頭辞の状態が入る。展開されたか、ビームから
             落ちた時点で手放す。
    parent:  この接頭辞の1つ前。根は None。

    配分は**保持しない**。親をたどって読み出すので、系譜が生まない配分を名乗る
    ことができない。parent と payload は repr から外してある（前者はエラーの1行に
    系譜全体を印字し、後者はギガバイトのテンソルを印字する）。
    """

    x: object = None
    score: object = None
    payload: object = field(default=None, repr=False)
    parent: object = field(default=None, repr=False)

    @classmethod
    def root(cls, payload=None):
        """歩き始めの空の接頭辞。

        始まりはこれ1本であって B 本ではない。層0 ではどの接頭辞も空の接頭辞
        なので、B 本置けば同じ層を同じ入力で B 回測り、同じ子を B 組作ることに
        なる。
        """
        return cls(payload=payload)

    @property
    def is_root(self):
        return self.parent is None

    def lineage(self):
        """根からこの接頭辞までを、層の順に並べたもの。"""
        chain, entry = [], self
        while entry is not None:
            chain.append(entry)
            entry = entry.parent
        chain.reverse()
        return chain

    @property
    def allocation(self):
        """層ごとの x（層0 が先頭）。系譜から導き、保存しない。"""
        return tuple(entry.x for entry in self.lineage()[1:])

    @property
    def depth(self):
        """決め終わった層数 = いま居る層。"""
        return len(self.lineage()) - 1

    def extend(self, x, score, payload=None):
        """1層ぶん決めて、この接頭辞を続ける子。"""
        return BeamEntry(x=x, score=score, payload=payload, parent=self)


class TopBeam:
    """1つの層の子のうち、良い方から ``width`` 本を、測った端から集める。

    ``insert`` はビームから出ていった物（いま入れた物のこともある）を返し、その
    前に ``on_evict`` を呼ぶ。呼び出し側が隠れ状態を捨てる場所は1箇所だけになり、
    まだビームに居る状態を捨てることができない。

    順序は全順序で決定的である — スコア、次にその子自身の x、次に到着順。同点は
    机上の話ではなく、Top-K=0 の層や、ちょうど変換できてしまう層では実際に起きる。
    どの候補も1トークンあたり同じ A 個の expert を走らせるので、どちらの同点崩し
    も目的関数からは任意である。x の小さい方を先にするのは、親を展開する順序に
    左右されないぶんまだ安定だからで、残りを到着順で崩す。こうしてビームが
    ``sort`` の同点の扱いに依存しなくなる。
    """

    def __init__(self, width, on_evict=None):
        if not width >= 1:
            raise ValueError(f'幅 {width} のビームは何も残さない')
        self.width = width
        self.on_evict = on_evict
        self._items = []
        self._arrivals = 0
        self._seen = set()

    def __len__(self):
        return len(self._items)

    def insert(self, entry):
        """採点済みの子を1つ差し出す。押し出された子を返す（無ければ None）。"""
        if entry.score is None:
            raise ValueError(f'{entry} にスコアが無い。測っていない子は順位を持てない')
        allocation = entry.allocation
        if allocation in self._seen:
            # 1つの層の子は、親か x のどちらかが違う。同じ物が2度来るのは各
            # 接頭辞を1度だけ展開していない歩き方で、それは自分で言うより狭い
            # ビームである
            raise ValueError(
                f'配分 {allocation} はこのビームに既に差し出されている。'
                '1つの層の子は作りからして互いに異なる')
        self._seen.add(allocation)

        self._items.append(((entry.score, entry.x, self._arrivals), entry))
        self._arrivals += 1
        self._items.sort(key=lambda item: item[0])
        if len(self._items) <= self.width:
            return None
        _, evicted = self._items.pop()
        if self.on_evict is not None:
            self.on_evict(evicted)
        return evicted

    @property
    def entries(self):
        """生き残り。良い順。"""
        return [entry for _, entry in self._items]

    @property
    def best(self):
        return self._items[0][1] if self._items else None


def parent_ranks(entries, parents):
    """各生存者について (親が親ビームの何位か, その x)。

    層ごとの表が「どの行から来た子か」を言うために要る。同一性で照合するのは、
    親の違う2つの子が同じ x と同じスコアを持ちうるからで、dataclass の ``==``
    はそれを同じ物だと言ってしまう。
    """
    ranks = {id(parent): rank for rank, parent in enumerate(parents)}
    located = []
    for entry in entries:
        if id(entry.parent) not in ranks:
            raise ValueError(
                f'{entry} は渡された {len(parents)} 本のどれからも展開されて '
                'いない。別の層のものである')
        located.append((ranks[id(entry.parent)], entry.x))
    return located


class BeamSearch:
    """接頭辞オラクルの上を、幅 ``width`` で前から歩く。

    width:    各層で生き残る接頭辞の本数。コストもホストのメモリもこれに比例する。
    budget:   使ってよいオラクルのコスト上限（単位はオラクルが決める）。超えたら
              ``BudgetExceeded`` を投げる。既定は上限なし。
    log:      1行ずつの表示先。
    on_layer: 層が終わるたびに記録を渡す先。数時間かかる実行が層30 で落ちても
              層0〜29 が残るように、書き出しは呼び出し側が層ごとに行う。

    探索はモデルにも配分の意味にも触れない。触れるのは「どの状態から x を足すか」
    と「返ってきたスコアの大小」だけである。
    """

    name = 'beam'

    def __init__(self, width=4, n_active_total=N_ACTIVE, budget=None, log=None,
                 on_layer=None, lookahead=0):
        if not width >= 1:
            raise ValueError(f'--width は 1 以上（{width}）')
        if not lookahead >= 0:
            raise ValueError(f'--lookahead は 0 以上（{lookahead}）')
        self.width = width
        # 刈る前に何層先まで展開するか。0 は現行のビーム。子1つあたりの採点回数が
        # 候補数の lookahead 乗で増える（候補7・深さ1 なら1つの子につき7回）
        self.lookahead = lookahead
        self.n_active_total = n_active_total
        self.budget = budget
        self.log = log or (lambda message='': None)
        self.on_layer = on_layer
        self.records = []

    def allocation_name(self):
        return f'{self.name}{self.width}{self._depth_suffix()}'

    def _depth_suffix(self):
        return f'L{self.lookahead}' if self.lookahead else ''

    def search(self, oracle, n_layers):
        """層 0..n_layers-1 を決めて、最良の配分を返す。"""
        if oracle is None:
            raise ValueError('この探索はオラクルを要求する')
        if not n_layers >= 1:
            raise ValueError(f'{n_layers} 層は探索できない')
        n_active_total = getattr(oracle, 'n_active_total', self.n_active_total)

        self._log_header(oracle)
        self.records = []
        entries = [BeamEntry.root(oracle.root())]
        best = entries[0]
        try:
            for layer in range(n_layers):
                self._check_budget(oracle, best)
                entries, record = self._walk_layer(oracle, layer, entries,
                                                   n_layers)
                best = entries[0]
                self.records.append(record)
                if self.on_layer is not None:
                    self.on_layer(self.records)
        finally:
            # 予算切れでも、途中の失敗でも、手元の接頭辞は必ず手放す。隠れ状態は
            # 1本で系列数ぶんの大きさがあり、握ったまま抜けると呼び出し側から
            # 解放する手立てが無い
            self._release_all(oracle, entries)

        return Allocation(best.allocation, name=self.allocation_name(),
                          n_active_total=n_active_total).check_layers(n_layers)

    @staticmethod
    def _release_all(oracle, entries):
        for entry in entries:
            oracle.release(entry.payload)
            entry.payload = None

    def _check_budget(self, oracle, best):
        """層に入る前に予算を見る。

        層の途中では止めない。1層を測り切らないと、その層の子は互いに比べられず、
        中途半端に測った層はビームを狭めるだけだからである。したがって超過は
        1層ぶんまで出うる — 予算は「これ以上は新しい層を始めない」という意味で
        あって、コストの上限そのものではない。
        """
        if self.budget is None or oracle.spent < self.budget:
            return
        raise BudgetExceeded(
            f'オラクルのコストが {oracle.spent:g} {oracle.cost_unit} に達した'
            f'（予算 {self.budget:g}）。決まったのは '
            f'{len(best.allocation)} 層',
            prefix=best.allocation, spent=oracle.spent, budget=self.budget)

    def _walk_layer(self, oracle, layer, parents, n_layers):
        """親ビームの全員をこの層で展開し、次のビームを返す。

        深さが 0 でないときは、子を1つ作るたびにその先を展開して**先の最良**を
        取り、それを子の順位に使う。子自身のスコアは記録には残るが、順位は決めない。
        """
        def evict(entry):
            oracle.release(entry.payload)
            entry.payload = None

        collector = TopBeam(self.width, on_evict=evict)
        rows, dropped = [], []
        try:
            for rank, parent in enumerate(parents):
                row = []
                for x in oracle.candidates(layer):
                    result, child_state = oracle.extend(parent.payload, layer, x)
                    try:
                        ahead, ahead_rows = self._look_ahead(
                            oracle, layer, child_state, n_layers, self.lookahead)
                    except BaseException:
                        # まだビームに入れていない子は、この層の後始末からは
                        # 見えない。作った側で捨てる
                        oracle.release(child_state)
                        raise
                    ranked = result.score if ahead is None else ahead
                    entry = {'x': x, 'score': result.score,
                             'details': result.details}
                    if ahead is not None:
                        entry['ranked_score'] = ahead
                        entry['lookahead'] = ahead_rows
                    row.append(entry)
                    evicted = collector.insert(
                        parent.extend(x, ranked, payload=child_state))
                    if evicted is not None:
                        dropped.append(evicted.score)
                # 親の状態はここで手放す。展開済みの接頭辞は子が引き継いでいる
                evict(parent)
                rows.append({'parent': rank, 'children': row})
        except BaseException:
            # 途中で落ちたら、まだ展開していない親と、集めかけの子の両方を捨てる。
            # 呼び出し側の finally はこの層のビームを知らない
            self._release_all(oracle, parents)
            self._release_all(oracle, collector.entries)
            raise

        entries = collector.entries
        if not entries:
            raise ValueError(f'層 {layer} で候補が1つも測られなかった')
        located = parent_ranks(entries, parents)
        worst_kept = entries[-1].score
        # 枝刈りが実際に何を決めたか: 落ちた中の最良と、残った中の最悪の差。
        # 子はより良い物にしか席を譲らないので非負になる。ノイズの尺度より
        # 小さい margin は「この層の切り方は何も決めていない」という意味になる
        margin = min(dropped) - worst_kept if dropped else None

        record = {
            'layer': layer,
            'lookahead': self.lookahead,
            'rows': rows,
            'beam': [{'rank': rank, 'parent': parent, 'x': entry.x,
                      'score': entry.score, 'allocation': list(entry.allocation)}
                     for rank, (entry, (parent, _)) in enumerate(zip(entries, located))],
            'margin': margin,
            'worst_kept': worst_kept,
            'best_dropped': None if margin is None else margin + worst_kept,
            'spent': oracle.spent,
            'cost_unit': oracle.cost_unit,
        }
        self._log_layer(record)
        return entries, record

    def _look_ahead(self, oracle, layer, state, n_layers, depth):
        """``state`` から先を ``depth`` 層ぶん展開し、(最良のスコア, 内訳) を返す。

        先の層で作った状態はここで全部手放す。持ち帰るのは数だけである — 先読みは
        「この子を残すか」を決めるためのもので、先の層はあとで本番の展開が
        もう一度通るからである（そのぶん採点は重複する。深さの代金はここに出る）。

        展開する層が残っていないとき（最終層の子）は ``(None, None)`` を返し、
        呼び出し側は子自身のスコアで順位を付ける。最終層の順位が子自身のスコアで
        付くことは、探索が返す score が接頭辞そのものの値であるために要る。
        """
        if depth < 1 or layer + 1 >= n_layers:
            return None, None
        rows, best = [], None
        for x in oracle.candidates(layer + 1):
            result, child = oracle.extend(state, layer + 1, x)
            try:
                deeper, _ = self._look_ahead(oracle, layer + 1, child, n_layers,
                                             depth - 1)
            finally:
                oracle.release(child)
            score = result.score if deeper is None else deeper
            rows.append({'x': x, 'score': score})
            best = score if best is None else min(best, score)
        return best, rows

    def _log_header(self, oracle):
        """表の列がどの x なのかを1行で出す。

        候補は層ごとに変わりうるので、層0 のものだと明示する。列の意味が無いと、
        読み手は星の位置から逆算するしかない。
        """
        candidates = oracle.candidates(0)
        if self.lookahead:
            self.log(f'深さ {self.lookahead}: 表の数はその子自身のスコア、'
                     '* と最善は先読みの最良で決めている')
        header = (f'{"層":>5} {"親":<4}'
                  + ' '.join(f'{f"x={x}":>10} ' for x in candidates)
                  + ' 最善（層0 の候補。* は次のビームへ残った子）')
        self.log(header)
        self.log('-' * 60)

    def _log_layer(self, record):
        survived = {(row['parent'], row['x']) for row in record['beam']}
        for row in record['rows']:
            cells = ' '.join(
                f'{child["score"]:>10.3e}'
                + ('*' if (row['parent'], child['x']) in survived else ' ')
                for child in row['children'])
            best = min(row['children'],
                       key=lambda child: child.get('ranked_score', child['score']))
            self.log(f'{record["layer"]:>5} {row["parent"]:<4}{cells}  '
                     f'x={best["x"]}')
        for row in record['beam']:
            self.log(f'{"":>5} {"":<4}   #{row["rank"]} {row["score"]:>10.3e}  '
                     f'親 {row["parent"]}  '
                     f'{",".join(str(x) for x in row["allocation"])}')
        if record['margin'] is not None:
            self.log(f'{"":>5} {"":<4}   margin {record["margin"]:>10.3e}  '
                     f'(落選 {record["best_dropped"]:.3e} / '
                     f'残留 {record["worst_kept"]:.3e})')


class GreedySearch(BeamSearch):
    """幅1のビーム = 各層でその場の最善を取る貪欲な歩き方。

    別実装ではない。「ビームが貪欲を上回った」を、2本のコードの差ではなく幅の差
    として言えるようにするために、同じコードを幅1で走らせる。
    """

    name = 'greedy'

    def __init__(self, n_active_total=N_ACTIVE, budget=None, log=None,
                 on_layer=None, lookahead=0):
        super().__init__(width=1, n_active_total=n_active_total, budget=budget,
                         log=log, on_layer=on_layer, lookahead=lookahead)

    def allocation_name(self):
        return f'{self.name}{self._depth_suffix()}'
