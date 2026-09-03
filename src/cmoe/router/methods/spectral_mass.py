"""方式7: expert の活性質量を、重み行列の低ランク近似から見積もる。

方式1〜5 はどれも「expert を代表する実在ニューロン1本」を動かしていた。その族の
中では、代表の選び直し（方式2・3・4）でも gain と offset の座標上昇（方式5）でも、
オラクル ``oracle_abs`` との Top-K 一致率は 0.15 で頭打ちになる（report/10）。
オラクルが読んでいるのは

    S_e(x) = Σ_{i∈e} |silu(g_i·x) · (u_i·x)|

という **expert 内の全ニューロンにわたる和**であり、1本の行がその和の順位を
決められないのは標本数の問題であって、選び方の問題ではない。

この方式は代表を捨て、和そのものを近似する。Cauchy–Schwarz の上界

    S_e(x) ≤ ‖silu(G_e x)‖ · ‖U_e x‖

を使う。行が互いに素な方向を向いているとき、この上界は S_e に比例した量になる
（比例定数は expert によらないので、Top-K の順位は保たれる）。右辺は2次形式の
積なので、``G_eᵀG_e`` と ``U_eᵀU_e`` の低ランク近似で好きな精度まで安く作れる。

2つの近似を入れている。

* **silu を行の重みに畳む。** ``Σ_i silu(a_i)² = Σ_i σ(a_i)² a_i²`` なので、
  σ(a_i)² を校正データ上の平均 ``p_i`` で置き換えると
  ``‖diag(√p) G_e x‖²`` という2次形式に戻る。ふだん負に居るニューロンはここで
  小さくなり、低ランクの予算がそちらへ流れない。
* **校正データで白色化してから分解する。** 近似したいのは行列そのものではなく
  ``E_x[‖W x‖²]`` なので、``x`` の2次モーメントで重みを付けた特異値分解を取る。

推論で増えるのは expert あたり ``2·r·hidden`` の積和だけである（r=32・
Llama-2-7B の S3A3E8 で、活性化する expert の計算量の約 1.3%）。学習は無く、
基底は特異値分解1回、行の重みは校正データ上の平均1回で決まる。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RANK = 32


STATISTICS_TOKENS = 32768


def flatten(z, hidden_size, limit=STATISTICS_TOKENS):
    """fit の z を [トークン, hidden] にし、統計に使う本数まで間引く。

    ここで読むのは4096×4096 の2次モーメントと行ごとの平均という、どちらも
    トークン数に対して速く収束する量である。全 131,072 位置を通しても値は
    小数点以下で動かない一方、時間は層あたり数秒増える。間引きは等間隔で、
    seed にも順序にも依らない。
    """
    if z is None:
        raise ValueError('spectral_mass は fit データを要求する')
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[1] != hidden_size:
        raise ValueError(
            f'fit_z の最終次元が {flat.shape[1]}、{hidden_size} のはず')
    if flat.shape[0] > limit:
        flat = flat[::max(1, flat.shape[0] // limit)][:limit]
    return flat


class SpectralMassRouter(nn.Module):
    """低ランクの2次形式2枚で expert の活性質量を見積もり、Top-K を選ぶ。

    基底は ``[n_routed, rank, hidden]`` を1枚の行列に畳んで持つ。推論では
    行列積1回で全 expert の射影が出る。

    値は float32 で作る。Top-K が読むのは **expert 間の score の差**で、それは
    score そのものの大きさに比べて小さい。bf16 の丸め幅がその差と同じ桁になると、
    順位を決めるのが質量ではなく丸めになる。
    """

    def __init__(self, gate_basis, up_basis, topk):
        super().__init__()
        if gate_basis.dim() != 3 or up_basis.dim() != 3:
            raise ValueError('基底は [experts, rank, hidden] のはず')
        if gate_basis.shape != up_basis.shape:
            raise ValueError(
                f'gate と up の基底の形が違う: {tuple(gate_basis.shape)} と '
                f'{tuple(up_basis.shape)}')
        n_routed, rank, hidden_size = gate_basis.shape
        if not 0 <= topk <= n_routed:
            raise ValueError(f'topk={topk} は routed expert {n_routed} 個に対して範囲外')

        self.dim = hidden_size
        self.topk = topk
        self.rank = rank
        self.n_routed = n_routed
        # 実在ニューロンの行ではないことを、診断と記録へ伝える
        self.representative_indices = None
        self.register_buffer(
            'projection',
            torch.cat([gate_basis.reshape(-1, hidden_size),
                       up_basis.reshape(-1, hidden_size)]).float().contiguous())

    def routing_scores(self, x):
        """[tokens, n_routed] の質量の見積もり。"""
        x = x.reshape(-1, self.dim)
        projected = F.linear(x.float(), self.projection)
        projected = projected.square().reshape(
            x.shape[0], 2, self.n_routed, self.rank).sum(dim=3)
        # 上界は2つのノルムの積。順位しか読まないので平方根は落とせるが、
        # 診断が相関を取るので質量と同じ次数のまま返す
        return (projected[:, 0] * projected[:, 1]).sqrt()

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        if self.topk == 0:
            indices = torch.empty((x.shape[0], 0), dtype=torch.long,
                                  device=x.device)
            return x.new_ones(indices.shape), indices
        indices = self.routing_scores(x).topk(self.topk, dim=1).indices
        return x.new_ones(indices.shape), indices


@torch.no_grad()
def silu_row_weights(gate_rows, z, chunk_size=8192):
    """gate 行ごとの ``√E[σ(g·x)²]``。silu の非対称性を行の大きさへ移す。"""
    total = None
    count = 0
    for start in range(0, z.shape[0], chunk_size):
        chunk = z[start:start + chunk_size].to(gate_rows.device, torch.float32)
        value = torch.sigmoid(F.linear(chunk, gate_rows)).square_().sum(
            dim=0, dtype=torch.float32)
        total = value if total is None else total + value
        count += chunk.shape[0]
    if not count:
        raise ValueError('fit データが空')
    return (total / count).sqrt()


@torch.no_grad()
def second_moment(z, hidden_size, device, chunk_size=8192):
    """校正データの ``E[x xᵀ]``（中心化しない）。

    塊ごとの積は float32 で取り、足し込みだけ float64 にする。この行列は
    このあと 1e-4 の対角ゆらぎを足してコレスキーに掛けるだけなので、
    塊ごとの丸め（相対 1e-5 程度）はそこに埋もれる。全体を float64 で
    掛けると、この機械では層あたり数秒がここだけに乗る。
    """
    moment = torch.zeros(hidden_size, hidden_size, dtype=torch.float64,
                         device=device)
    count = 0
    for start in range(0, z.shape[0], chunk_size):
        chunk = z[start:start + chunk_size].to(device, torch.float32)
        moment += (chunk.T @ chunk).double()
        count += chunk.shape[0]
    if not count:
        raise ValueError('fit データが空')
    return moment / count


@torch.no_grad()
def whitened_basis(weight, chol, rank):
    """``E[‖W x‖²]`` を最もよく説明する rank 本の方向（特異値を掛けた形）。

    ``x = L u`` と置くと ``‖W x‖² = ‖(W L) u‖²`` で、u は白色である。したがって
    ``W L`` の右特異ベクトルの上位が、データの上での寄与が大きい順になる。
    その方向を x 空間へ戻して返す。

    ``chol`` は float32 で受け取る。コレスキーだけ float64 で取るのは分解が
    正定値性に敏感だからで、そのあとの積と分解にその精度は要らない。
    """
    projected = weight @ chol
    _, singular, right = torch.linalg.svd(projected, full_matrices=False)
    take = min(rank, right.shape[0])
    directions = torch.linalg.solve_triangular(
        chol.T, right[:take].T, upper=True).T
    return directions * singular[:take, None]


@torch.no_grad()
def plain_basis(weight, rank):
    """白色化しない素の特異値分解の上位 rank 本。"""
    _, singular, right = torch.linalg.svd(weight.float(), full_matrices=False)
    take = min(rank, right.shape[0])
    return right[:take] * singular[:take, None]


class SpectralMassMethod:
    """低ランクの活性質量ルーター。"""

    name = 'spectral_mass'
    training_free = True
    requires_source_weights = True
    requires_fit_z = True

    rank = DEFAULT_RANK
    # ``spectral_mass:64`` と書いたときに値が入る属性
    variant_attribute = 'rank'
    # silu を行の重みに畳む段。切ると素の ‖G_e x‖·‖U_e x‖ になる
    silu_weighting = True
    # 校正データで白色化してから分解する段。切ると重みだけで基底が決まる
    whiten = True

    @torch.no_grad()
    def build(self, context, baseline):
        if context.gate_weight is None or context.up_weight is None:
            raise ValueError('spectral_mass は dense の gate/up 重みを要求する')
        rank = int(self.rank)
        if rank < 1:
            raise ValueError(f'rank は 1 以上 (受け取った値: {self.rank!r})')
        device = context.gate_weight.device
        # 校正データを読むのは白色化と silu の行の重みだけ。どちらも切った
        # 変種では、fit_z に一度も触らない
        needs_z = self.whiten or self.silu_weighting
        z = flatten(context.fit_z, context.hidden_size) if needs_z else None

        chol = None
        if self.whiten:
            moment = second_moment(z, context.hidden_size, device)
            jitter = 1e-4 * float(torch.diagonal(moment).mean())
            chol = torch.linalg.cholesky(
                moment + jitter * torch.eye(context.hidden_size,
                                            dtype=torch.float64,
                                            device=device)).float()
            del moment

        # 行の重みは routed 全体を1回で通す。expert ごとに z を読み直すと、
        # 同じ量を expert の数だけ数えることになる
        routed = [index for group in context.routed_groups for index in group]
        row_weights = None
        if self.silu_weighting:
            rows = torch.tensor(routed, dtype=torch.long, device=device)
            row_weights = silu_row_weights(
                context.gate_weight.index_select(0, rows).float(), z)

        gate_basis, up_basis = [], []
        offset = 0
        for group in context.routed_groups:
            rows = torch.tensor(list(group), dtype=torch.long, device=device)
            if not rows.numel():
                raise ValueError('routed expert が空')
            gate = context.gate_weight.index_select(0, rows).float()
            up = context.up_weight.index_select(0, rows).float()
            if row_weights is not None:
                gate = gate * row_weights[offset:offset + rows.numel(), None]
                offset += rows.numel()
            if self.whiten:
                gate_basis.append(whitened_basis(gate, chol, rank))
                up_basis.append(whitened_basis(up, chol, rank))
            else:
                gate_basis.append(plain_basis(gate, rank))
                up_basis.append(plain_basis(up, rank))
            del gate, up

        router = SpectralMassRouter(torch.stack(gate_basis),
                                    torch.stack(up_basis), context.topk)
        return router.to(device)


class PlainSpectralMassMethod(SpectralMassMethod):
    """白色化を切った方式7。基底が校正データに依らない。

    白色化は「校正データの上で ``E[‖W x‖²]`` を最もよく説明する方向」を選ぶが、
    低ランクではその方向自体が校正の分布に強く依存する。切ると基底は重みだけ
    から決まるので、校正と評価の分布が違っても同じ近似になる（silu の行の
    重みだけは校正データを読む）。校正の中での近似精度と、校正の外への
    移りやすさのどちらを取るかを分ける行である。
    """

    name = 'spectral_plain'
    whiten = False


class WeightOnlySpectralMassMethod(SpectralMassMethod):
    """校正データを1ビットも読まない方式7。

    silu の行の重みも切るので、基底も倍率も dense の重みだけから決まる。
    ``spectral_mass`` → ``spectral_plain`` → ``spectral_weight`` と辿ると、
    校正データへの依存が2段で外れる。校正の中での近似精度が落ちる代わりに、
    校正と評価の分布が違っても振る舞いが変わらない。
    """

    name = 'spectral_weight'
    whiten = False
    silu_weighting = False
