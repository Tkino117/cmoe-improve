"""配分オラクルの土台: 接頭辞を1層ずつ前へ進める。

どのオラクルも同じ道具立てを要る — 接頭辞の隠れ状態から attention を進め、
活性を測り、x ごとに層を分割して MoE を作り、その出力を次の層へ渡す。違うのは
**何を測って点にするか**だけである。共通部分を ``LayerWalk`` に置き、指標は
``measure`` の1メソッドに閉じる。

進め方は組み立て役（``assemble.Converter``）と同じ関数を通る。統計は
``carve.profile.profile_layer``、次層への伝播は ``moe.forward_chunked`` で、
どちらも1実装しかない。探索が測った軌道と、あとで変換が載せる軌道が黙って
分かれないための取り決めである。

**モデルは書き換えない。** 層を MoE に差し替えて測り、あとで戻す、ということを
しない。attention をすでに別で進めてあるので、変換後の層の出力は
``moe(z) + residual`` そのものであり、モデルを触らずに次の層の入力が作れる。
戻し忘れという失敗の形が最初から無く、同じ層の別の x を測るのも並びの問題では
なくなる。

**alloc は router を import しない。** 層に載る MoE を作るのは外から渡される
``LayerFactory`` で、分割規則もルーター方式もオラクルからは見えない。両方を
同時に使う知識は assemble.py にしか置かない、という依存の向きをここで守る。

メモリの都合は2つの軸に分かれている。

``state_device``（既定はホスト）は**接頭辞の状態の置き場所**である。ビームは
2B+1 本の状態を同時に抱え、1本が [系列数, seqlen, hidden] なので n=64 なら
1本 1 GiB、B=4 で 9 GiB になる。これをカードに置くと、そこへ dense の読み出しと
候補の読み出しが加わる。移送元が payload をホストに置いていたのはこのためで、
既定を引き継いでいる。

``batch_chunk`` は**1回の前向きで何系列を進めるか**である。渡さなければ層の
入口で全系列をカードへ載せ、z も residual も層の側に残る（速い）。渡すと系列を
分けて進め、z も H もホスト側に残る。後者では、重みを使う計算 — ルーターの
呼び出しと層ローカル指標の行列積・集計 — を塊ごとに重みの側へ渡して走らせる
（``token_chunk``）。どちらもトークンごとに閉じた計算なので、分けても値は
変わらない。
"""

from dataclasses import dataclass, field
from typing import Protocol

import torch

from cmoe.alloc.base import ScoreResult
from cmoe.carve.profile import (hidden_activations, profile_layer,
                                select_positions)
from cmoe.moe.modules import forward_chunked


@dataclass
class CarvedLayer:
    """x を1つ決めたときに、その層へ載る物。

    ``moe`` は配備できる実体そのもの（ルーターも expert 重みも入っている）。
    ``partition`` はどのニューロンがどの expert かで、層ローカル指標はこれを
    読む。
    """

    n_shared: int
    topk: int
    moe: object
    partition: object


class LayerFactory(Protocol):
    """x → その層に載る ``CarvedLayer``。中身はオラクルから見えない。"""

    def __call__(self, dense, rates, markers, n_shared, topk) -> CarvedLayer:
        ...


@dataclass
class PrefixState:
    """接頭辞が次の層へ渡すもの。探索から見て中身は不透明。

    hidden: 次の層への入力。``release`` されたあとは None になる。
    depth:  決め終わった層数。
    score:  その接頭辞のスコア。積み上げ型の指標（層ローカル誤差）が、次の層で
            足し込む先として読む。
    pinned: キャリブレーション入力そのもの。オラクルが持ち主なので捨てない。
    """

    hidden: object
    depth: int = 0
    score: float = 0.0
    pinned: bool = False


@dataclass
class LayerProfile:
    """1つの接頭辞 × 1つの層で、x に依存しないものすべて。

    捕捉と活性統計は x を変えても同じなので、その層の候補 7 個で共有する。
    ``h_true`` は真の中間活性 H（層ローカル指標だけが読む）で、要求されるまで
    作らない。
    """

    layer: int
    dense: object
    z: object
    residual: object
    rates: object
    markers: object
    # 活性を数えた位置だけを取り出した z。絞らないときは z そのもの。分割規則
    # は統計と同じ位置を見なければならないので、分割へ渡すのはこちらである
    profile_z: object = None
    h_true: object = field(default=None, repr=False)


class LayerWalk:
    """1つのモデルの上で、接頭辞を層で進める。

    保持するのは「どの接頭辞のどの層まで測ったか」だけで、beam も配分も持たない
    （それは探索側のものである）。
    """

    def __init__(self, adapter, inputs, layer_factory, n_experts,
                 n_active_total=6, k_act=10, profiling_norm=True,
                 batch_chunk=None, state_device='cpu', profile_mask=None,
                 score_weights=None, choices=None):
        self.adapter = adapter
        self.inputs = inputs
        self.layer_factory = layer_factory
        self.n_experts = n_experts
        self.n_active_total = n_active_total
        self.k_act = k_act
        self.profiling_norm = profiling_norm
        self.batch_chunk = batch_chunk
        self.state_device = torch.device(state_device)
        # 活性を数える位置（``TokenSet.scored_mask()``）。None なら全位置。
        # 組み立て役と同じものを渡さないと、探索が測った分割とあとで載る分割が
        # 別物になる
        self.profile_mask = profile_mask
        # オラクルが位置ごとに掛ける重み（``TokenSet.position_weights()``）。
        # None なら全位置が等しい。**分割を決める ``profile_mask`` とは別物で
        # ある** — こちらが決めるのは「候補をどの位置で採点するか」であって、
        # 「どの位置の活性でニューロンを切り分けるか」ではない。2つを別々に
        # 動かせることが、形の効果と位置の効果を分ける実験点になる
        self.score_weights = score_weights
        # 選択肢どうしを比べる目的関数が要る対応（``alloc.base.ChoiceScoring``）。
        # None なら、そういう目的関数は組めない。**位置ごとの重みとは別物で
        # ある** — あちらは位置に閉じた量の平均の取り方で、こちらは「1つの点が
        # K 本の系列にまたがる」という構造そのものである
        self.choices = choices
        if choices is not None:
            expected = tuple(inputs.hidden.shape[:2])
            if tuple(choices.keep.shape) != expected:
                raise ValueError(
                    f'選択肢の印は {tuple(choices.keep.shape)}、校正入力は '
                    f'{expected} — 対応していない')
        if score_weights is not None:
            expected = tuple(inputs.hidden.shape[:2])
            if tuple(score_weights.shape) != expected:
                raise ValueError(
                    f'重みは {tuple(score_weights.shape)}、校正入力は '
                    f'{expected} — 対応していない')
            if not float(score_weights.sum()) > 0:
                raise ValueError('重みが全位置で 0。採点する位置が無い')
        self._flat_weights = None
        # 1接頭辞 × 1層ぶんだけ持つ。beam は1つの親の子を続けて測るので、
        # これで捕捉と統計は層ごとに1回になる。親の状態への参照を握っている
        # あいだは、その状態が解放されないことも保証される。
        self._cache = None

    @property
    def n_layers(self):
        return self.adapter.n_layers

    @property
    def device(self):
        return self.adapter.device

    def flat_score_weights(self):
        """重みを [トークン] に潰したもの。重みが無ければ None。

        位置ごとの量（層ローカル指標のトークンごとの和、KL のトークンごとの値）
        は、どれも ``reshape(-1)`` と同じ並びで出てくる。並べ替えの規則が1つ
        しかないことが、重みとトークンの対応が崩れない理由である。
        """
        if self.score_weights is None:
            return None
        if self._flat_weights is None:
            self._flat_weights = self.score_weights.reshape(-1)
        return self._flat_weights

    def candidates(self, layer):
        """その層で試せる x。

        上限は A（1トークンあたりに走る expert 数）で、同時に x < N でなければ
        routed expert が残らない。層に依らないが、層ごとに候補を変える探索の
        余地を残して引数を取る。
        """
        return tuple(x for x in range(self.n_active_total + 1)
                     if x < self.n_experts)

    def root(self):
        """何も決めていない接頭辞。キャリブレーション入力そのもの。"""
        self.adapter.to_device()
        return PrefixState(hidden=self.inputs.hidden.to(self.state_device),
                           depth=0, score=0.0, pinned=True)

    def release(self, state):
        """この状態はもう読まれない。隠れ状態と、その上で作った捕捉を捨てる。

        キャッシュを落とすのは pinned な状態（キャリブレーション入力）について
        も行う。捨てないのは入力そのものだけで、その上で作った z・residual・
        真の H は層1本ぶんの大きさがあり、握ったままだと層0 の捕捉が実行の
        最後まで残る。
        """
        if state is None:
            return
        if self._cache is not None and self._cache[0] is state:
            self._cache = None
        if not state.pinned:
            state.hidden = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def profile(self, state, layer):
        """接頭辞が層 ``layer`` へ渡す入力を捕まえ、活性統計を作る。"""
        if self._cache is not None:
            cached_state, cached_layer, profile = self._cache
            if cached_state is state and cached_layer == layer:
                return profile
        self._cache = None
        if state.hidden is None:
            raise ValueError(
                f'深さ {state.depth} の接頭辞は隠れ状態を解放済み。'
                '展開する前に release された')
        if state.depth != layer:
            raise ValueError(
                f'深さ {state.depth} の接頭辞を層 {layer} で展開しようとしている')

        dense = self.adapter.dense_ffn(layer)
        # 状態はホストに置くが、測るのは層の側である。バッチを分けないときは
        # ここでカードへ載せ、z も residual もそのまま層の側に残す（分けるときは
        # ``forward_attention`` が塊ごとに載せ、結果を状態と同じ側へ戻す）
        hidden = (state.hidden if self.batch_chunk is not None
                  else state.hidden.to(self.device))
        z, residual = self.adapter.forward_attention(
            layer, hidden, self.inputs.attention_mask,
            self.inputs.position_ids, batch_chunk=self.batch_chunk)
        del hidden
        profile_z = select_positions(z, self.profile_mask)
        rates, markers = profile_layer(
            dense, profile_z, k_act=self.k_act, normalize=self.profiling_norm,
            batch_chunk=self.batch_chunk, device=self.device)
        profile = LayerProfile(layer=layer, dense=dense, z=z, residual=residual,
                               rates=rates, markers=markers,
                               profile_z=profile_z)
        self._cache = (state, layer, profile)
        return profile

    @torch.no_grad()
    def carve(self, profile, x):
        """候補 x の層を作る。Top-K は A - x で、A は全候補で同じ。"""
        if x not in self.candidates(profile.layer):
            raise ValueError(
                f'x={x} は層 {profile.layer} の候補 '
                f'{self.candidates(profile.layer)} に無い')
        carved = self.layer_factory(profile.dense, profile.rates, profile.markers,
                                    x, self.n_active_total - x,
                                    z=profile.profile_z, layer=profile.layer)
        if carved.n_shared != x or carved.topk != self.n_active_total - x:
            raise ValueError(
                f'x={x} を渡したのに x={carved.n_shared} Top-K={carved.topk} '
                'の層が返ってきた')
        return carved

    @torch.no_grad()
    def propagate(self, profile, carved):
        """この候補を載せた層の出力 = 次の層への入力。

        戻り値は ``state_device`` に置く。ビームは 2B+1 本の状態を同時に抱える
        ので、カードに残すと n=64・B=4 で 9 GiB がそこに乗る（移送元がここを
        ホストに置いていたのと同じ理由）。
        """
        hidden = forward_chunked(carved.moe, profile.z, profile.residual,
                                 batch_chunk=self.batch_chunk, device=self.device)
        return PrefixState(hidden=hidden.to(self.state_device),
                           depth=profile.layer + 1)

    @torch.no_grad()
    def true_activations(self, profile):
        """真の中間活性 H を [トークン, ニューロン] で返す。

        プロファイル用の h（入力と重みを正規化したもの）ではない。層ローカル
        指標が測るのは「実際に走らなかった活性」なので、正規化した方を読むと
        別の量になる。1層につき1回だけ作り、その層の候補で共有する。
        """
        if profile.h_true is not None:
            return profile.h_true
        z = profile.z
        step = self.batch_chunk or z.shape[0]
        rows = []
        for start in range(0, z.shape[0], step):
            chunk = z[start:start + step].to(self.device)
            h = hidden_activations(profile.dense, chunk, normalize=False)
            rows.append(h.to(z.device))
            del h, chunk
        h = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        profile.h_true = h.reshape(-1, h.shape[-1])
        return profile.h_true


def apply_weights(per_token, weights):
    """位置ごとの量に、オラクルの重みを掛ける。重みが無ければそのまま返す。

    重みが無い経路では引数のテンソルをそのまま返す（掛け算も確保もしない）ので、
    既存の測定の数は1ビットも動かない。
    """
    if weights is None:
        return per_token
    if weights.shape[0] != per_token.shape[0]:
        raise ValueError(
            f'重みは {weights.shape[0]} 位置、測ったのは {per_token.shape[0]} '
            'トークン — 対応していない')
    return per_token * weights.to(dtype=per_token.dtype, device=per_token.device)


class PrefixOracleBase:
    """接頭辞オラクルの共通部分。指標は ``measure`` にだけ書く。

    コストは ``cost_unit`` の単位で積む。単位が違うオラクルどうしのコストは
    比べられない — 「同じ予算で何が取れたか」を言えるのは、同じ単位を報告する
    オラクルの間だけである。
    """

    name = 'prefix'
    cost_unit = 'oracle_calls'
    # 層ローカル指標は「走らせなかった活性」を測るので、走らせないものが無い
    # 設定では何も測らない。それを名乗るオラクルはこれを True にする
    needs_routing = False

    def __init__(self, walk):
        self.walk = walk
        if self.needs_routing and walk.n_active_total >= walk.n_experts:
            # A >= N ではどの x でも Top-K = A-x が routed 数 N-x 以上になり、
            # ルーターが全部を選ぶ。スコアは全候補で 0 になり、ビームは同点崩し
            # で並べた配分を「成功」として返す
            raise ValueError(
                f'A={walk.n_active_total} は N={walk.n_experts} 以上。どの候補も '
                f'routed を全部走らせるので、{self.name} は全候補で 0 を返し、'
                '何も測らない（A < N にするか、suffix_kl を使う）')
        self._spent = 0.0
        self._calls = 0

    @property
    def n_layers(self):
        return self.walk.n_layers

    @property
    def n_active_total(self):
        return self.walk.n_active_total

    @property
    def spent(self):
        return self._spent

    @property
    def calls(self):
        return self._calls

    def candidates(self, layer):
        return self.walk.candidates(layer)

    def root(self):
        return self.walk.root()

    def release(self, state):
        self.walk.release(state)

    @torch.no_grad()
    def extend(self, state, layer, x):
        """接頭辞に層 ``layer`` の x を足し、(スコア, 次の状態) を返す。"""
        profile = self.walk.profile(state, layer)
        carved = self.walk.carve(profile, x)
        child = self.walk.propagate(profile, carved)
        result = self.measure(profile, carved, state, child)
        child.score = result.score
        self._spent += result.cost
        self._calls += 1
        del carved
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result, child

    def measure(self, profile, carved, state, child) -> ScoreResult:
        """この接頭辞のスコア。小さいほど良い。

        返すのは**接頭辞全体**のスコアであって、足した1層のぶんではない。層を
        またいでどう積むか（足し合わせるのか、測り直すのか）は指標の性質で
        決まり、探索はそれを知らない。
        """
        raise NotImplementedError


def score_allocation(oracle, allocation, log=None):
    """決まった配分ベクトル1本を、接頭辞オラクルで頭から採点する。

    探索が返した配分を測り直すため、また層ごとの内訳を見るための道具。
    ``ScoreResult`` を返し、``details['per_layer']`` に各層のスコアが入る。
    """
    state = oracle.root()
    rows = []
    result = None
    for layer, x in enumerate(allocation):
        result, child = oracle.extend(state, layer, x)
        # details も残す。積み上げ型の指標では、その層**単独**の値がここにしか
        # 無い（score は接頭辞の合計なので、内訳は引き算でしか出せない）
        rows.append({'layer': layer, 'x': x, 'score': result.score,
                     'cost': result.cost, 'details': result.details})
        if log is not None:
            log(f'  層 {layer:>2}: x={x} score={result.score:.6e}')
        oracle.release(state)
        state = child
    oracle.release(state)
    if result is None:
        raise ValueError('配分が空。採点する層が1つも無い')
    return ScoreResult(score=result.score, cost=oracle.spent,
                       cost_unit=oracle.cost_unit,
                       details={'per_layer': rows,
                                'allocation': list(allocation),
                                'calls': oracle.calls})
