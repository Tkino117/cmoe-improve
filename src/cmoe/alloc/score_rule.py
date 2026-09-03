"""層別スカラー s_ℓ を配分ベクトル x_ℓ に落とす、規則型の共通の型。

**提案手法ではない。対照の器である。** ExpertWeaver の式(5)(6) が持っている形

    α_ℓ = α_max − (α_max − α_min) · s_ℓ ,   x_ℓ = round(α_ℓ · N)

を、``s_ℓ`` の作り方から切り離したものである。EW は s に「CV > τ なニューロン
の割合」を入れる（``alloc.ew_rule``）。OWL は「外れ値の割合」を入れる
（``alloc.owl``）。**規則型の対照が2本以上になると、負けた理由が「その1本の
統計が悪い」なのか「層別スカラー → 単調写像という方式そのもの」なのかを分けて
書けるようになる** — それがこの器を切り出した理由である。

自由度は2つだけに畳んである。

* ``direction``  … ``'desc'`` なら s が大きい層ほど x が小さい（= routing が
  多い）。``'asc'`` はその逆。**どちらが正しいかは統計ごとに違い、事前には
  決まらない**ので引数にしてある。EW は desc（特化した層ほど shared を減らす）。
  OWL も desc に対応する（外れ値の多い層ほど「削らない」＝ routing で広く
  カバーする）が、これは移送側の解釈なので、校正で選び直せるようにしてある。
* ``(α_min, α_max)`` … 写像の値域。EW の既定は [0.2, 0.7]。

``normalize`` は s を [0, 1] に均す規則である。EW の s はもともと割合なので
``'none'`` でそのまま使えるが、OWL の外れ値比率は 1e-3 の桁で、生のまま入れると
α がほぼ α_max に張り付いて**規則が一様配分に潰れる**。OWL 自身も層別スパース率
を [S−λ, S+λ] に再スケールしているので、min-max で値域を使い切るのが原典に
沿う。潰れたかどうかは ``detail['degenerate']`` に出る。

クリップ（``x`` を [0, A] に収める）と、その層数を返すのは ``ew_rule`` と同じ
約束である。A=4 で上限を超えた層が何本あったかは結果の読みに効く。
"""

from cmoe.alloc.ew_rule import _round_half_up

DIRECTIONS = ('desc', 'asc')
NORMALIZERS = ('minmax', 'none')


def normalized(scores, normalize='minmax'):
    """s を [0, 1] に均す。

    ``minmax`` は層をまたいだ最小・最大で割る。全層が同じ値なら 0 を返す
    （その場合どんな α レンジでも一様配分になる — 潰れたことは呼び側が
    ``degenerate`` で知る）。
    """
    if normalize not in NORMALIZERS:
        raise ValueError(f'normalize は {NORMALIZERS} のどれか（{normalize!r}）')
    values = [float(s) for s in scores]
    if normalize == 'none':
        return values
    low, high = min(values), max(values)
    if not high > low:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def rule_allocation(scores, n_experts, n_active, alpha_min, alpha_max,
                    direction='desc', normalize='minmax'):
    """層別スカラー → ``(values, detail)``。

    ``detail`` は ``ew_rule.ew_allocation`` と同じ鍵を持つ（``ratios`` の中身が
    正規化後の s になる）。同じ読み方ができるように揃えてある。
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction は {DIRECTIONS} のどれか（{direction!r}）')
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError(
            f'α は 0 ≤ α_min ≤ α_max ≤ 1（{alpha_min}, {alpha_max}）')
    ratios = normalized(scores, normalize)
    if direction == 'asc':
        ratios = [1.0 - r for r in ratios]

    alphas, raw, values = [], [], []
    for ratio in ratios:
        alpha = alpha_max - (alpha_max - alpha_min) * ratio
        x = _round_half_up(alpha * n_experts)
        alphas.append(alpha)
        raw.append(x)
        values.append(min(max(x, 0), n_active))
    return values, {
        'direction': direction,
        'normalize': normalize,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'raw_scores': [float(s) for s in scores],
        'ratios': ratios,
        'alphas': alphas,
        'x_before_clip': raw,
        'n_clipped': sum(1 for a, b in zip(raw, values) if a != b),
        'mean_x': sum(values) / len(values) if values else 0.0,
        'n_routing_layers': sum(1 for x in values if x < n_active),
        'degenerate': len(set(values)) <= 1,
    }
