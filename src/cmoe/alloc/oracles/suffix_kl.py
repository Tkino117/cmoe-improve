"""接頭辞を、モデル自身の予測がどれだけ動いたかで採点する。

CMoE-ref の ``xsearch/downstream.py`` と ``scripts/beam_search.py`` の目的関数の
移送。層ローカル指標（``alloc.oracles.local_error``）はその層の出力空間の話で
止まり、その誤差を後続の層が吸収するのか増幅するのかを言えない。ここでは残りの
層を実際に走らせて、出力分布の差を測る:

    KL(接頭辞) = mean_t KL( P_dense(t) ‖ P_接頭辞(t) )    nats / トークン

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
def kl_divergence(reference_logits, candidate_logits, token_chunk=TOKEN_CHUNK):
    """mean_t KL( P_reference(t) ‖ P_candidate(t) )、nats / トークン。

    全位置が対象で、先頭のトークンも入る（ずらしは無い — これは予測の採点では
    なく、同じ入力に対する2つの分布の比較である）。引数の順序がそのまま
    divergence の向きで、対称ではない。

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

    total = 0.0
    for start in range(0, n_tokens, token_chunk):
        stop = min(start + token_chunk, n_tokens)
        log_p = reference[start:stop].to(torch.float32).log_softmax(dim=-1)
        log_q = candidate[start:stop].to(dtype=torch.float32,
                                         device=log_p.device).log_softmax(dim=-1)
        total += float((log_p.exp() * (log_p - log_q)).sum(dtype=torch.float64))
    kl = total / n_tokens

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
                0, state.hidden, self.walk.inputs, batch_chunk=self.batch_chunk)
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
            0, state.hidden, self.walk.inputs, batch_chunk=self.batch_chunk)
        floor, _ = kl_divergence(reference, repeated, self.token_chunk)
        del repeated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return floor

    @torch.no_grad()
    def measure(self, profile, carved, state, child):
        logits = self.walk.adapter.forward_suffix(
            profile.layer + 1, child.hidden, self.walk.inputs,
            batch_chunk=self.batch_chunk)
        kl, n_tokens = kl_divergence(self.dense_logits, logits, self.token_chunk)
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
