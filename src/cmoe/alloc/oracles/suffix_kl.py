"""接頭辞を、モデル自身の予測がどれだけ動いたかで採点する。

CMoE-ref の ``xsearch/downstream.py`` と ``scripts/beam_search.py`` の目的関数の
移送。層ローカル指標（``alloc.oracles.local_error``）はその層の出力空間の話で
止まり、その誤差を後続の層が吸収するのか増幅するのかを言えない。ここでは残りの
層を実際に走らせて、出力分布の差を測る:

    KL(接頭辞) = Σ_t w_t KL( P_dense(t) ‖ P_接頭辞(t) ) / Σ_t w_t   nats / トークン

重み w は既定では全位置で等しく、そのとき素の平均になる（移送元と同じ）。
``--scored-weight`` を渡すと、答え部分の位置が目的関数のどれだけを占めるかを
決められる。1問1系列の校正では位置の8割が埋めなので、そこを 0 にしないと、
探索は埋めの上で dense に近い配分を選ぶ。**重みが 0 の位置の読み出しは持たない**
ので、確保も数える位置ぶんで済む。

目標は**dense モデル**の分布で、何も変換していない時点で1度だけ読み、実行中
ずっと変えない。層も親も違う接頭辞どうしが同じ物差しの上に乗るのは、これが
固定されているからである。

**接頭辞のスコアが何であって、何でないか。** 層 ℓ を測るとき、モデルは 0..ℓ が
変換済みで、ℓ+1..最終層は **dense のまま**である。つまりスコアが答えているのは
「残りが dense のままなら、この接頭辞はいくら損か」であって、配備されるモデルに
dense の層は1つも無い。この食い違いは消えるのではなく後ろへ送られ、最後の層で
ちょうど閉じる — そこには dense の suffix が残っていないので、最終層で測った
スコアは実在するモデルの KL そのものである。それより前はすべて「どの接頭辞を
生かしておくか」のための当て推量である。

読み出しの決定性には床がある。同じ dense モデルを2回読んで KL を取ると、
カーネルの非決定性のぶんだけゼロにならない。``nondeterminism_floor()`` がそれで、
これより小さい差で2つの配分を区別することはできない。
"""

import torch

from cmoe.alloc.base import ScoreResult
from cmoe.alloc.oracles.base import PrefixOracleBase

# log_softmax 1回あたりのトークン数。fp32 の [トークン, 語彙] を2枚同時に置くのが
# ここで一番大きい確保になるので、塊で進めて2枚が全長で存在しないようにする。
TOKEN_CHUNK = 1024


@torch.no_grad()
def kl_divergence(reference_logits, candidate_logits, token_chunk=TOKEN_CHUNK,
                  weights=None):
    """Σ_t w_t KL( P_reference(t) ‖ P_candidate(t) ) / Σ_t w_t、nats / トークン。

    ずらしは無い — これは予測の採点ではなく、同じ入力に対する2つの分布の比較で
    ある。引数の順序がそのまま divergence の向きで、対称ではない。

    ``weights`` が None なら全位置が等しい（これが既定で、既存のすべての測定が
    通る経路である）。重みを渡すと、その重み付き平均になる。重みは相対値でよく、
    合計で割るので大きさには依らない。

    重みのある枝は per-token の和を先に取る。**重みが無いときはそちらへ入らない**
    — 足す順序が変われば最後の数ビットが動き、この指標が候補に順序を付けるのは
    小数第4位より下だからである。

    log_softmax は両方 fp32 で、トークンの塊ごとに進める。
    """
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f'形が違う: reference {tuple(reference_logits.shape)} / '
            f'candidate {tuple(candidate_logits.shape)}')
    vocab = reference_logits.shape[-1]
    reference = reference_logits.reshape(-1, vocab)
    candidate = candidate_logits.reshape(-1, vocab)
    n_tokens = reference.shape[0]
    if n_tokens == 0:
        raise ValueError('比べるトークンが無い')
    if weights is not None:
        if weights.shape[0] != n_tokens:
            raise ValueError(
                f'重みは {weights.shape[0]} 位置、読み出しは {n_tokens} トークン '
                '— 対応していない')
        weight_total = float(weights.sum(dtype=torch.float64))
        if not weight_total > 0:
            raise ValueError('重みの合計が 0。採点する位置が無い')

    total = 0.0
    for start in range(0, n_tokens, token_chunk):
        stop = min(start + token_chunk, n_tokens)
        log_p = reference[start:stop].to(torch.float32).log_softmax(dim=-1)
        log_q = candidate[start:stop].to(dtype=torch.float32,
                                         device=log_p.device).log_softmax(dim=-1)
        terms = log_p.exp() * (log_p - log_q)
        if weights is None:
            total += float(terms.sum(dtype=torch.float64))
        else:
            per_token = terms.sum(dim=-1, dtype=torch.float64)
            chunk = weights[start:stop].to(dtype=torch.float64,
                                           device=per_token.device)
            total += float((per_token * chunk).sum(dtype=torch.float64))
        del terms
    if weights is None:
        kl = total / n_tokens
    else:
        kl = total / weight_total
        n_tokens = int((weights > 0).sum())

    # KL は Jensen より非負。負が出るのは和の桁落ち（項の fp32 イプシロンで
    # 抑えられ、この閾値よりはるかに小さい）か誤りで、NaN は比較に失敗して
    # 順序を当てずっぽうに決めずに例外になる
    if not kl > -1e-6:
        raise ValueError(f'KL={kl} が負か非数')
    return kl, n_tokens


class SuffixKLOracle(PrefixOracleBase):
    """残りの層を dense のまま走らせて、出力分布の差を測る。

    層ローカル指標よりずっと高価で（子1つにつき、自分の層と後続の層すべての
    forward）、そのぶん配備されるモデルの振る舞いに近い。CMoE-ref のビーム探索は
    この目的関数で走っており、``alloc/presets.py`` の ``beam`` はその結果である。
    """

    name = 'suffix_kl'
    cost_unit = 'layer_forwards'

    def __init__(self, walk, token_chunk=TOKEN_CHUNK, batch_chunk=None):
        super().__init__(walk)
        self.token_chunk = token_chunk
        # 読み出しは [bsz, seq, 語彙] と大きいので、捕捉とは別に分割幅を持てる
        self.batch_chunk = batch_chunk if batch_chunk is not None else walk.batch_chunk
        # 重みが 0 の位置は KL に1度も入らないので、読み出しも持たない。1問1系列の
        # 校正では埋めが位置の8割を占め、[位置, 語彙] の確保がそのぶん丸ごと消える
        weights = walk.flat_score_weights()
        self.keep = None if weights is None else (weights > 0).reshape(
            walk.score_weights.shape)
        self.weights = None if weights is None else weights[weights > 0]
        self._dense_logits = None

    @property
    def dense_logits(self):
        """dense モデルの読み出し。1度だけ作って持ち続ける。

        オラクルはモデルを書き換えないので、これはいつ取っても同じである
        （元実装が「最初の変換より前に」と念を押していたのは、あちらが層を
        差し替えながら測っていたためである）。
        """
        if self._dense_logits is None:
            state = self.walk.root()
            self._dense_logits = self.walk.adapter.forward_suffix(
                0, state.hidden, self.walk.inputs, batch_chunk=self.batch_chunk,
                keep=self.keep)
        return self._dense_logits

    @torch.no_grad()
    def nondeterminism_floor(self):
        """同じ dense 読み出しを2回取って比べた KL。

        重みも forward も同じなので、2つを隔てているのはこの機械のカーネルの
        非決定性だけである。これより小さい差で2つの配分を区別することはできない。
        """
        reference = self.dense_logits
        state = self.walk.root()
        repeated = self.walk.adapter.forward_suffix(
            0, state.hidden, self.walk.inputs, batch_chunk=self.batch_chunk,
            keep=self.keep)
        floor, _ = kl_divergence(reference, repeated, self.token_chunk,
                                 self.weights)
        del repeated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return floor

    @torch.no_grad()
    def measure(self, profile, carved, state, child):
        logits = self.walk.adapter.forward_suffix(
            profile.layer + 1, child.hidden, self.walk.inputs,
            batch_chunk=self.batch_chunk, keep=self.keep)
        kl, n_tokens = kl_divergence(self.dense_logits, logits, self.token_chunk,
                                     self.weights)
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 自分の層と、そのあとに走らせた層。探索どうしを同じ単位で比べるための数
        cost = float(self.walk.n_layers - profile.layer)
        return ScoreResult(
            score=kl, cost=cost, cost_unit=self.cost_unit,
            details={'kl': kl, 'n_tokens': n_tokens, 'layer': profile.layer,
                     'x': carved.n_shared, 'topk': carved.topk,
                     'dense_suffix_layers': self.walk.n_layers - profile.layer - 1})
