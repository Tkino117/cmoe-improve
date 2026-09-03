"""方式8: 固定 Top-K をやめ、平均だけを守って expert をトークン間で配り直す。

方式1〜7 はどれも「どのトークンにも K 個」という決め打ちを共有していた。
オフラインの探査（``experiments/19_spectral_router/dynamic_k.py``）が示したのは、
**その決め打ちの方が、K 個をどう選ぶかより大きく損をしている**ことである。
同じ平均 K のまま、score の絶対値でトークンをまたいで配り直すと、層の出力誤差
``‖y_dense − y_sel‖/‖y_dense‖`` は3層平均で

    現行 CMoE 固定K   0.4147
    現行 CMoE 可変K   0.4104   ← 追加の積和は1回も無い
    オラクル 固定K    0.3930   ← report/18 で acc を +0.0221 上げたもの
    低ランク 可変K    0.3661
    オラクル 可変K    0.3357

と動く。固定 K のオラクルより、可変 K の配備できるルーターの方が誤差が小さい。

理由は、score の**絶対値**が「このトークンが routed 側をどれだけ要るか」を持って
いることにある。softmax も、トークンごとの最大値で割る規格化（D2DMoE の
dynamic-k がそうする）も、その情報をちょうど捨てる。実際、探査ではトークンごとに
規格化する配り方は固定 K と変わらなかった。

配り方は score のしきい値1個で表せる。しきい値は層ごとに1つのスカラーで、校正
データの上で平均が K になるように決める。推論ではトークンごとに独立に決まるので、
自己回帰でも使える（(トークン × expert) を並べ直す必要は無い）。

**予算は平均でしか守られない。** 1トークンだけを見れば K を超えることも下回る
こともある。評価データの上で実際に何個走ったかは ``selection`` に残す — 平均が
揃っていなければ、この比較は成立しない。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmoe.router.methods.spectral_mass import (DEFAULT_RANK, flatten,
                                               plain_basis, silu_row_weights)


def baseline_scores(router, x):
    """現行 CMoE ルーターの、softmax を通す**前**の score。

    softmax はトークンごとに和を1にするので、トークン間で score の大きさを
    比べられなくする。可変 K が読むのはまさにその大きさなので、ここでは
    通さない。固定 Top-K では softmax が順位を変えないため、この量が現行
    ルーターの選択をそのまま決めている。
    """
    return (router.classifier(x) * F.silu(router.gate(x))).abs().float()


class ThresholdRouter(nn.Module):
    """score がしきい値を超えた expert を全部選ぶ。選択は 0/1 で返す。

    ``minimum`` を立てると、どのトークンにも上位 ``minimum`` 個は無条件で配る。
    しきい値だけだと score が一様に小さいトークンで routed が1つも立たなく
    なるので、その床を持たせるための引数である（既定は 0 = 床なし）。
    """

    dynamic_topk = True

    def __init__(self, source, threshold, topk, n_routed, minimum=0):
        super().__init__()
        self.source = source
        self.dim = source.dim
        # 予算としての K。実際に走る数はトークンごとに違い、平均でこれに合う
        self.topk = topk
        self.n_routed = n_routed
        self.minimum = int(minimum)
        self.representative_indices = None
        self.register_buffer('threshold',
                             torch.tensor(float(threshold), dtype=torch.float32))
        # 走った expert 数とトークン数の積算。校正で決めたしきい値が評価データ
        # でも予算を守っているかは、この方式の比較が成立する条件そのものなので、
        # 校正の外での実測を残せるようにしておく
        self.register_buffer('load', torch.zeros(2, dtype=torch.float64),
                             persistent=False)
        # 何個走ったかの分布。平均が予算に合っていても、中身が「ほとんどの
        # トークンが0個で、一部が5個」なら別の話なので、形も残す
        self.register_buffer('load_histogram',
                             torch.zeros(n_routed + 1, dtype=torch.float64),
                             persistent=False)

    def routing_scores(self, x):
        return self.source.routing_scores(x)

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        scores = self.routing_scores(x)
        mask = scores >= self.threshold
        if self.minimum:
            mask = mask | torch.zeros_like(mask).scatter_(
                1, scores.topk(self.minimum, dim=1).indices, True)
        # カードの上で足す。ここで float() を取ると1層ごとに同期が入る
        counts = mask.sum(dim=1)
        self.load[0] += counts.sum()
        self.load[1] += mask.shape[0]
        self.load_histogram += torch.bincount(
            counts, minlength=self.n_routed + 1)
        return x.new_ones(mask.shape), mask


def reset_load(routers):
    """層ごとのルーターが数えている「走った expert 数」を0に戻す。"""
    for router in routers:
        if isinstance(router, ThresholdRouter):
            router.load.zero_()
            router.load_histogram.zero_()


def load_histogram(routers):
    """走った routed expert 数の分布（層をまたいで足した割合）。可変 K だけ。

    **層ごとに routed expert の数が違ってよい。** 配分 x が層ごとに変わる構成
    では routed は N - x_ℓ 本で、層ごとに数え上げの幅（0..N-x_ℓ）が変わる。
    短い方を右にゼロで伸ばしてから足す — その層は定義上そこに票を持てないので、
    ゼロは「数えていない」ではなく「起こり得ない」を表す。

    したがって非一様な配分では、この分布の k 番目は**層をまたいで意味が揃って
    いない**（ある層の「5個全部」は別の層の「5個中5個」ではない）。層ごとの
    割合が要るときは ``ThresholdRouter.load_histogram`` を直接読む。
    """
    histograms = [router.load_histogram for router in routers
                  if isinstance(router, ThresholdRouter)]
    if not histograms:
        return None
    width = max(histogram.numel() for histogram in histograms)
    total = torch.zeros(width, dtype=histograms[0].dtype,
                        device=histograms[0].device)
    for histogram in histograms:
        total[:histogram.numel()] += histogram
    if not float(total.sum()):
        return None
    return (total / total.sum()).tolist()


def mean_load(routers):
    """評価の間に実際に走った routed expert 数の平均（層をまたいだ平均）。

    可変 Top-K のルーターが1つも無ければ ``None``。予算を守っているかを
    校正の外で確かめるための唯一の数字なので、記録に残す。

    **平均を取るのは可変 K のルーターを持つ層だけである。** Top-K=0 の層は
    そもそも ``ThresholdRouter`` を持たない（``Converter`` が基準ルーターを
    渡す）ので、ここには入らない。したがって非一様な配分では、この値を
    「A − 平均 x」と比べてはいけない — 比べる相手は **K>0 の層だけで平均した
    予算** である。
    """
    values = [float(router.load[0] / router.load[1])
              for router in routers
              if isinstance(router, ThresholdRouter) and float(router.load[1])]
    if not values:
        return None
    return sum(values) / len(values)


class BaselineScoreRouter(nn.Module):
    """現行 CMoE ルーターを、score を名乗るだけの形に包んだもの。

    しきい値の側は「score を出す何か」しか要求しない。包むことで、可変 K の
    実装が方式1 の中身（代表ニューロンの2本の行）を知らずに済む。
    """

    def __init__(self, router):
        super().__init__()
        self.router = router
        self.dim = router.dim
        self.topk = router.topk

    def routing_scores(self, x):
        return baseline_scores(self.router, x)


class SpectralScoreRouter(nn.Module):
    """方式7 と同じ低ランクの質量見積もりを、score としてだけ出すもの。

    基底は**白色化しない**素の特異値分解で取る。白色化した基底は校正データの
    上でだけ score が持ち上がる（その data で選んだ方向なので当然である）ため、
    校正で決めたしきい値が評価データでずれる — 探査では平均 K が 3.00 から
    2.69 へ落ちた。固定 Top-K は score の絶対値を読まないのでこのずれに気付か
    ないが、可変 K は絶対値そのものを読む。
    """

    def __init__(self, gate_basis, up_basis, topk):
        super().__init__()
        n_routed, rank, hidden_size = gate_basis.shape
        self.dim = hidden_size
        self.topk = topk
        self.rank = rank
        self.n_routed = n_routed
        self.register_buffer(
            'projection',
            torch.cat([gate_basis.reshape(-1, hidden_size),
                       up_basis.reshape(-1, hidden_size)]).float().contiguous())

    def routing_scores(self, x):
        x = x.reshape(-1, self.dim)
        projected = F.linear(x.float(), self.projection).square().reshape(
            x.shape[0], 2, self.n_routed, self.rank).sum(dim=3)
        return (projected[:, 0] * projected[:, 1]).sqrt()


@torch.no_grad()
def calibrate_threshold(source, z, topk, n_routed, minimum, device,
                        chunk_size=8192):
    """校正データの上で、平均の選択数が topk になるしきい値を決める。

    床（``minimum``）で無条件に出るぶんはしきい値の対象から外してから数える。
    戻り値は (しきい値, 校正の上での平均選択数)。
    """
    if not 0 <= minimum <= topk:
        raise ValueError(f'床 {minimum} は予算 {topk} の中に無い')
    values = []
    for start in range(0, z.shape[0], chunk_size):
        chunk = z[start:start + chunk_size].to(device)
        scores = source.routing_scores(chunk)
        if minimum:
            floor = scores.topk(minimum, dim=1).values[:, -1:]
            scores = scores.masked_fill(scores >= floor, -math.inf)
        values.append(scores.flatten())
    values = torch.cat(values)
    n_tokens = z.shape[0]
    remaining = int(round((topk - minimum) * n_tokens))
    finite = values[values.isfinite()]
    if remaining <= 0:
        return math.inf, float(minimum)
    if remaining >= finite.numel():
        return -math.inf, float(n_routed)
    threshold = float(finite.topk(remaining).values[-1])
    selected = int((finite >= threshold).sum())
    return threshold, minimum + selected / n_tokens


class DynamicTopKMethod:
    """現行 CMoE の score を、固定 Top-K ではなくしきい値で使う。

    ルーターの演算も重みも1ビットも変えない。変えるのは「何個選ぶか」だけで、
    追加の積和は無い。
    """

    name = 'dynamic_cmoe'
    training_free = True
    requires_fit_z = True
    # どのトークンにも最低これだけ配る
    minimum = 0
    variant_attribute = 'minimum'

    def source(self, context, baseline):
        return BaselineScoreRouter(baseline)

    @torch.no_grad()
    def build(self, context, baseline):
        device = baseline.gate.weight.device
        z = flatten(context.fit_z, context.hidden_size)
        source = self.source(context, baseline)
        threshold, mean_k = calibrate_threshold(
            source, z, context.topk, context.n_routed, self.minimum, device)
        router = ThresholdRouter(source, threshold, context.topk,
                                 context.n_routed, minimum=self.minimum)
        router.selection = {'threshold': threshold, 'fit_mean_k': mean_k}
        return router.to(device)


class DynamicSpectralMethod(DynamicTopKMethod):
    """方式7 の低ランク score を、しきい値で使う。

    固定 Top-K の方式7 が校正データで基底を選んでよかったのは、順位しか読まな
    かったからである。しきい値は絶対値を読むので、ここでは白色化を切る。
    """

    name = 'dynamic_spectral'
    requires_source_weights = True
    rank = DEFAULT_RANK
    variant_attribute = 'rank'

    @torch.no_grad()
    def source(self, context, baseline):
        if context.gate_weight is None or context.up_weight is None:
            raise ValueError('dynamic_spectral は dense の gate/up 重みを要求する')
        device = context.gate_weight.device
        z = flatten(context.fit_z, context.hidden_size)
        routed = [index for group in context.routed_groups for index in group]
        rows = torch.tensor(routed, dtype=torch.long, device=device)
        row_weights = silu_row_weights(
            context.gate_weight.index_select(0, rows).float(), z)

        gate_basis, up_basis, offset = [], [], 0
        for group in context.routed_groups:
            rows = torch.tensor(list(group), dtype=torch.long, device=device)
            gate = context.gate_weight.index_select(0, rows).float()
            gate = gate * row_weights[offset:offset + rows.numel(), None]
            offset += rows.numel()
            gate_basis.append(plain_basis(gate, int(self.rank)))
            up_basis.append(plain_basis(
                context.up_weight.index_select(0, rows).float(), int(self.rank)))
        return SpectralScoreRouter(torch.stack(gate_basis),
                                   torch.stack(up_basis), context.topk).to(device)
