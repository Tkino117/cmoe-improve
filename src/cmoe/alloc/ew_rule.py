"""ExpertWeaver の層別配分規則（arXiv:2602.15521 §3.2）を、この土俵に移したもの。

**提案手法ではない。対照である。** ExpertWeaver 全体（多タスク校正・balanced
K-Means による expert 構築・gate セントロイドのルーター）の再現ではなく、
**配分を決める式3本だけ**を取り出して、現行 CMoE の分割・ルーターの上に載せる。
比べたいのは「配分の決め方」なので、それ以外は対照と1ビットも変えない。

移せる理由は、両者のパラメトリゼーションが一致していることにある。EW は
「全 N_se,ℓ 個の共有 + ルーターが選ぶ上位 k−N_se,ℓ 個、トークンあたり合計 k 個」
と置いており、これは ``x_ℓ + K_ℓ = A`` そのものである。**予算はどちらの側でも
不変**なので、配分だけを差し替えても土俵が崩れない。

規則は3段（式 4・5・6）:

1. ``a_ij`` = サンプル j のトークン平均 ``|Swish(g_i·x)|``。ニューロン i の
   活性プロファイル
2. ``CV_i`` = サンプル方向の変動係数。``r_ℓ`` = ``CV_i > τ`` なニューロンの割合
3. ``α_ℓ = α_max − (α_max−α_min)·r_ℓ``、``x_ℓ = round(α_ℓ·N)``

**原論文と違えた点を3つ明記しておく。**

* **サンプルの単位。** EW の1サンプルは「1タスクの few-shot 例1本」で、計 240
  本ある。サンプル**間**の変動がタスク間の変動になる、というのが CV が特化を
  測れる理由である。ここは校正トークンの総量を他の行と揃えたうえで、窓を
  ``sample_tokens`` トークンに切って1サンプルとする。2048 トークン窓1本を
  1サンプルにすると、平均が窓の中で多タスク性を潰してしまい、CV が EW の
  想定より1桁小さく出て τ がどのニューロンも拾わなくなる（``result_logs/
  ew_probe_{slimpajama,flanv2}_seed0``。flanv2 は窓内でタスクを交互に並べる
  ので、こちらの方が縮退が強い）。128 トークンなら 16 系列 × 16 = M=256 で、
  EW の 240 とほぼ同じ本数になる。
* **絶対値を取らない（既定）。** EW 式は ``mean(Swish(x W_gate))`` と符号付きで
  書かれている。Swish は負側が −0.278 で下げ止まるので、めったに活性しない
  ニューロンはトークン平均が 0 近傍になり、CV が大きく出る。**これは事故では
  なく規則が使っている性質である** — 「CV が高い＝特化」はこの挙動に乗って
  いる。``use_abs=True`` にすると平均に正の下駄が入って CV 分布が潰れ、
  τ=0.6 がどのニューロンも拾わなくなる（``result_logs/ew_probe_*_seed0`` の
  4本はすべてこれで縮退した）。EW §3.3 が「絶対値付き平均活性化スコア」を
  使うのは shared **の選択**であって、CV の定義ではない。
* **粒度と上限。** EW の既定は expert 粒度 64、α∈[0.2, 0.7]。ここは N=8 なので
  ``round(α·8)`` は 2〜6 に落ちる。A=6（スパース率25%）なら収まるが、A=4（50%）
  では上限を超えるので ``[0, A]`` にクリップする。**クリップした層数は結果の
  読みに効くので必ず記録する**（クリップが多ければ、この規則は 50% で
  ``uniform{A}`` に潰れている）。

CV は τ・α・A のどれにも依存しない。GPU を使うのは ``layer_gate_profile`` の
1回だけで、そこから先（しきい値の格子も、スパース率の違いも）は CPU で足りる。
"""

import math

import torch
import torch.nn.functional as F

# EW §4.1 の既定ハイパーパラメータ
DEFAULT_TAU = 0.6
DEFAULT_ALPHA_MIN = 0.2
DEFAULT_ALPHA_MAX = 0.7


@torch.no_grad()
def layer_gate_profile(dense, z, sample_tokens=None, use_abs=False,
                       batch_chunk=None, device=None):
    """FFN 入力 z から ``[サンプル, ニューロン]`` の ``a_ij`` を作る。

    ``carve/profile.py`` の ``hidden_activations`` を使わないのは、あちらが up
    側を掛けた ``h`` を返すのに対し、EW が読むのは **gate 側だけ**の
    ``Swish(g_i·x)`` だからである。正規化もしない（EW の式に無い）。

    ``sample_tokens`` は1サンプルの長さ。``None`` なら系列1本を1サンプルと
    する。**この値が CV の大きさを直接決める** — 長く取るほど平均が変動を潰し、
    CV が小さくなって τ が何も拾わなくなる。モジュール冒頭の但し書きを見ること。

    ``use_abs`` は EW 式から外れる側の枝である。既定の ``False`` が原論文どおり。
    """
    if z.dim() != 3:
        raise ValueError(f'[bsz, seq, hidden] のはず（{tuple(z.shape)}）')
    seq_len = z.shape[1]
    if sample_tokens is not None and seq_len % sample_tokens:
        raise ValueError(
            f'系列長 {seq_len} が sample_tokens={sample_tokens} で割り切れない。'
            '端を捨てるとサンプルごとに平均する本数が変わる')
    step = batch_chunk or z.shape[0]
    rows = []
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step]
        if device is not None:
            chunk = chunk.to(device)
        gate = dense.act_fn(F.linear(chunk, dense.gate_proj.weight))
        gate = gate.to(torch.float32)
        if use_abs:
            gate = gate.abs()
        if sample_tokens is not None:
            gate = gate.reshape(-1, sample_tokens, gate.shape[-1])
        rows.append(gate.mean(dim=1).to('cpu'))
        del gate, chunk
    return torch.cat(rows, dim=0)


def neuron_cv(profile, eps=1e-12):
    """``[サンプル, ニューロン]`` から、ニューロンごとの変動係数。

    分母は平均の**絶対値**。符号付きプロファイル（既定）では平均が負にも
    なりうるので、そのまま割ると CV の符号が反転して τ の比較が壊れる。

    標準偏差は不偏（``unbiased=True``）。EW は指定していないが、M が小さいので
    どちらを取ったかは書き残す必要がある。
    """
    if profile.dim() != 2:
        raise ValueError(f'[サンプル, ニューロン] のはず（{tuple(profile.shape)}）')
    if profile.shape[0] < 2:
        raise ValueError(
            f'サンプルが {profile.shape[0]} 本しかない。CV は 2 本以上要る')
    mean = profile.mean(dim=0)
    std = profile.std(dim=0, unbiased=True)
    return std / mean.abs().clamp_min(eps)


def _round_half_up(value):
    """0.5 を上へ。Python の ``round`` は偶数へ丸めるので使わない。

    ``α·N`` がちょうど .5 になる r は実在する（N=8・既定の α なら r=0.525）ので、
    ここが銀行丸めだと「規則の出力」が直感と食い違う。
    """
    return int(math.floor(value + 0.5))


def specialization_ratios(cvs, tau=DEFAULT_TAU):
    """式(4)。層ごとの特化比率 ``r_ℓ``。"""
    return [float((cv > tau).to(torch.float32).mean()) for cv in cvs]


def ew_allocation(cvs, n_experts, n_active, tau=DEFAULT_TAU,
                  alpha_min=DEFAULT_ALPHA_MIN, alpha_max=DEFAULT_ALPHA_MAX):
    """式(4)(5)(6)。層ごとの CV から配分ベクトルを出す。

    戻り値は ``(values, detail)``。``detail`` は層ごとの ``r_ℓ`` / ``α_ℓ`` /
    クリップ前の x と、クリップした層数を持つ。**クリップ数を返り値に入れて
    あるのは、それが結果の解釈に必要な事実だからである** — 呼び出し側が捨てる
    ことはできるが、知らずに済ませることはできないようにしてある。
    """
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError(
            f'α は 0 ≤ α_min ≤ α_max ≤ 1（{alpha_min}, {alpha_max}）')
    ratios = specialization_ratios(cvs, tau)
    alphas, raw, values = [], [], []
    for ratio in ratios:
        alpha = alpha_max - (alpha_max - alpha_min) * ratio
        x = _round_half_up(alpha * n_experts)
        alphas.append(alpha)
        raw.append(x)
        values.append(min(max(x, 0), n_active))
    detail = {
        'tau': tau,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'ratios': ratios,
        'alphas': alphas,
        'x_before_clip': raw,
        'n_clipped': sum(1 for a, b in zip(raw, values) if a != b),
        'mean_x': sum(values) / len(values) if values else 0.0,
        'n_routing_layers': sum(1 for x in values if x < n_active),
    }
    return values, detail
