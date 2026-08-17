"""L(x): 変換後の層が実際に出す出力誤差。その推定量ではない。

CMoE-ref の ``xsearch/output_error.py`` の移送。R(x)（``alloc.oracles.mass``）は
ニューロンを |h| や h^2 で重み付けるので、down projection の列の大きさも、
ニューロンどうしの打ち消しも見ていない。L(x) はその代理をやめて、当のものを
測る:

    L = Σ_t ‖ (h_t ⊙ missed_t) W_down^T ‖² / Σ_t ‖ h_t W_down^T ‖²

``missed_t`` はトークン t で MoE が**走らせない**ニューロン、つまり shared にも
選ばれた routed expert にも入らないものである。小さいほど良く、routed を全部
選べばちょうど 0 になる。

これが近似ではなく相対二乗誤差そのものになるのは、学習なしの経路に限った2つの
理由による。

* expert は dense の gate/up/down の行・列をそのまま切り出したものなので、
  expert が走らせるニューロンは dense と同じ h を計算する
* ``MoE.forward`` は選んだ expert の出力を routing weight で掛けるが、その重みは
  ちょうど 1.0 である（``Router.forward`` が返すのは
  ``1 + softmax_score * extra_scale``、``extra_scale`` はゼロ初期化で学習されない）

**前提**: ``extra_scale`` がゼロでなくなれば（ファインチューニング、ルーターの
変更）重みは 1 でなくなり、この数は黙って近似に戻る。

精度の規則も2つ。h は bf16 で、候補の順序を決める差は L の小数第4位に居る。
マスクも2つの行列積も fp32 で行い、TF32 は前後で落として戻す（fp32 の仮数を
ほとんど捨ててしまうため）。

接頭辞のスコアは層ごとの L の**和**である。
"""

from contextlib import contextmanager
import weakref

import torch

from cmoe.alloc.base import ScoreResult
from cmoe.alloc.oracles.base import PrefixOracleBase
from cmoe.alloc.oracles.mass import neuron_to_expert, router_selection

# 行列積1回あたりのトークン数。マスク後の H は fp32 なので、7B 規模
# （11008 ニューロン）で 4096 トークンなら 180 MiB で済む。和はトークンごとに
# 独立なので、分けても値は変わらない。
TOKEN_CHUNK = 4096


def token_chunks(n_tokens, step):
    """トークンを分けて進めるときの [start, stop) の並び。"""
    step = n_tokens if step is None else step
    return [(start, min(start + step, n_tokens))
            for start in range(0, n_tokens, max(step, 1))]


def missed_activations(h_chunk, indices_chunk, group_of, n_routed):
    """この塊の H から、MoE が走らせるニューロンをゼロにしたもの。fp32。

    必ず新しいテンソルを作る。``copy=True`` が無いと、すでに fp32 の H では
    ``.to(float32)`` が同じテンソルを返し、その場のマスクが、以降の候補が測られる
    はずの H を壊す。
    """
    runs = torch.zeros(h_chunk.shape[0], n_routed + 1, dtype=torch.bool,
                       device=h_chunk.device)
    runs[:, n_routed] = True              # shared は全トークンで走る
    if indices_chunk.numel():
        runs.scatter_(1, indices_chunk.to(torch.int64), True)
    missed = h_chunk.to(torch.float32, copy=True)
    missed.masked_fill_(runs[:, group_of], 0.0)
    return missed


@contextmanager
def exact_fp32_matmul():
    """fp32 の行列積を TF32 ではなく fp32 で走らせる。

    TF32 は fp32 の仮数 23 ビットのうち 10 ビットしか残さない。この指標が順序を
    付けたい候補どうしは L の小数第4位で違うので、プロセス内の他のコードが立てた
    ``allow_tf32`` が、その順序を黙ってノイズの中に沈める。例外で抜けるときも
    含めて元に戻す。
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def output_energy(h, w32, token_chunk=TOKEN_CHUNK, device=None):
    """トークンごとの ‖ h_t W_down^T ‖²。dense の側、L の分母。"""
    per_token = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
    for start, stop in token_chunks(h.shape[0], token_chunk):
        chunk = h[start:stop].to(device) if device is not None else h[start:stop]
        dense = chunk.to(torch.float32) @ w32.T
        per_token[start:stop] = dense.square_().sum(dim=1).to(h.device)
        del chunk, dense
    return per_token


@torch.no_grad()
def missed_energy(h, w32, group_of, indices, n_routed, token_chunk=TOKEN_CHUNK,
                  device=None):
    """トークンごとの ‖ (h_t ⊙ missed_t) W_down^T ‖²。L の分子。"""
    per_token = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
    for start, stop in token_chunks(h.shape[0], token_chunk):
        chunk = h[start:stop].to(device) if device is not None else h[start:stop]
        rows = (indices[start:stop].to(device) if device is not None
                else indices[start:stop])
        missed = missed_activations(chunk, rows, group_of, n_routed)
        left_out = missed @ w32.T
        per_token[start:stop] = left_out.square_().sum(dim=1).to(h.device)
        del chunk, rows, missed, left_out
    return per_token


class LocalErrorOracle(PrefixOracleBase):
    """層ローカル: その層の出力誤差 L(x)。後続の層は走らせない。

    分母は dense の FFN だけで決まるので、1つの層の候補すべてで使い回す。
    """

    name = 'local_error'
    cost_unit = 'layer_ffn_evals'
    # 走らせなかった活性の作る誤差を測るので、A >= N では何も測らない
    needs_routing = True

    def __init__(self, walk, token_chunk=TOKEN_CHUNK):
        super().__init__(walk)
        if token_chunk is not None and not token_chunk >= 1:
            raise ValueError(f'token_chunk は 1 以上か None（{token_chunk}）')
        self.token_chunk = token_chunk
        self._total = None

    def _total_energy(self, profile, h, w32, device):
        """分母。同じ層の候補で使い回す（次の層へ移ったら作り直す）。

        分母は dense の FFN だけで決まるので、1つの層の候補すべてが同じものを
        使う。キャッシュが持つのは**弱参照**である：強参照だと、``LayerWalk``
        が捕捉を手放したあともここが掴んでいて、その層の z・residual・真の H が
        次の層の途中まで生き残る（1層ぶんで n=64 なら数 GiB）。死んだプロファイル
        と同じアドレスに来た別のプロファイルを取り違える危険も、弱参照なら同時に
        消える。
        """
        if self._total is not None:
            reference, per_token = self._total
            if reference() is profile:
                return per_token
        per_token = output_energy(h, w32, self.token_chunk, device)
        self._total = (weakref.ref(profile), per_token)
        return per_token

    @torch.no_grad()
    def measure(self, profile, carved, state, child):
        h = self.walk.true_activations(profile)
        n_tokens, n_neurons = h.shape
        w_down = profile.dense.down_proj.weight
        if w_down.shape[1] != n_neurons:
            raise ValueError(
                f'down_proj は {w_down.shape[1]} ニューロン、H は {n_neurons}')

        # H はバッチを分けたときホストに残る。行列積は重みが載っている側で
        # 走らせ、トークンの塊だけを行き来させる — 分母は [トークン, ニューロン]
        # と大きく、CPU で回すと本番の規模で桁違いに遅い
        device = w_down.device
        group_of = neuron_to_expert(carved.partition, n_neurons, device)
        n_routed = len(carved.partition.routed_groups)
        indices = router_selection(carved.moe, profile.z, self.token_chunk)
        if indices.shape[0] != n_tokens:
            raise ValueError(
                f'ルーターが {indices.shape[0]} 行を返した（トークンは '
                f'{n_tokens} 個）')

        w32 = w_down.to(torch.float32)
        with exact_fp32_matmul():
            per_token_total = self._total_energy(profile, h, w32, device)
            per_token_missed = missed_energy(h, w32, group_of, indices, n_routed,
                                             self.token_chunk, device)
        del w32

        total = float(per_token_total.sum())
        if not total > 0:
            raise ValueError(f'dense FFN 出力の二乗ノルムが {total}。H が空か非有限')
        value = float(per_token_missed.sum() / per_token_total.sum())
        # 二乗和の比なので負にはならない。ここに来るのは H が非有限のときだけで、
        # NaN は比較に失敗して素通りせず例外になる
        if not value >= 0.0:
            raise ValueError(f'L={value} が負か非数。H がおそらく非有限')
        # L <= 1 は定理ではない。走らせた寄与と走らせなかった寄与が打ち消し合うと、
        # 取りこぼしのほうが元の出力より長くなりうる。実際の層で起きるとは
        # 考えにくいが起こり得るので、印を付けて先へ進む（32層の実行を層20 で
        # 落とす価値は無い）
        over_unity = not value <= 1.0

        return ScoreResult(
            score=state.score + value, cost=1.0, cost_unit=self.cost_unit,
            details={'l': value, 'over_unity': over_unity,
                     'missed_energy': float(per_token_missed.sum()),
                     'total_energy': total, 'n_tokens': n_tokens,
                     'layer': profile.layer, 'x': carved.n_shared,
                     'topk': carved.topk})
