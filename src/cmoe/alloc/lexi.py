"""LExI (arXiv:2509.02753) の層別配分を、この土俵に移す。

**提案手法ではない。対照である。** そして、提案手法と最も近い位置にいる対照で
ある — どちらも「層ごとの出力誤差を測って、予算のもとで配分を決める」形をして
いて、違うのは**誤差をどこまで通して測るか**だけになる。

LExI がやっているのは2段である。

1. **感度プロファイル。** 層 ℓ の active expert 数だけを動かし、**他の層は
   baseline のまま**にして、その層の出力の Frobenius ずれを測る。層ごとに
   独立な表 ``S[ℓ][k]`` ができる
2. **予算制約下の探索。** ``Σ_ℓ S[ℓ][k_ℓ]`` を、総 active expert 数の予算のもとで
   最小化する。原典はここを進化探索で解く

移送で変わるのは 2 だけである。**この土俵では x に予算制約が無い** — x が何で
あれ1トークンあたりに走る expert 数は A で一定だからである（``alloc.base``）。
制約が消えると ``Σ_ℓ S[ℓ][x_ℓ]`` は層ごとに分離するので、進化探索は
**層ごとの argmin** に厳密に退化する。原典の探索器を持ち込まないのはそのため
であって、手を抜いたからではない。逆に言えば、LExI がこの軸に持ち込む本質は
**「層ローカルの出力誤差を、他の層を baseline に置いたまま測る」**の一点である。

**原典から違えた点がもう1つある。** LExI は感度プロファイルを**無作為な合成入力**
で取る（data-free を売りにしている）。ここは他の行と同じ校正データの上で取る。
理由は2つで、(i) 実際の活性の方が合成入力より情報が多く、LExI を有利に扱うことに
なる、(ii) 校正を揃えないと「配分の決め方だけを比べている」が崩れる。原典から
外れた側なので必ず書き残す。

1 の側は ``alloc.oracles.local_error`` がそのまま使える。あれが測る

    L = Σ_t ‖ (h_t ⊙ missed_t) W_down^T ‖² / Σ_t ‖ h_t W_down^T ‖²

は、その層の出力の相対二乗誤差そのもの（近似ではない）で、LExI の Frobenius
ずれと同じものを正規化した形である。**baseline を dense に取る**のが移送の要で、
これは ``experiments/22_alloc_baselines/lexi_probe.py`` が層を dense のまま
進めることで実現している。探索が使う ``greedy`` は変換済みの接頭辞の上を進む
ので別物である — その差こそが「層ローカルか、後続まで通すか」の差になる。
"""


def argmin_allocation(table, n_active, prefer='low'):
    """層ごとの独立な argmin。``table`` は ``[層][x] -> L``。

    ``prefer`` は同点のときにどちらの x を取るか。``'low'`` は小さい x（= routing
    が多い側）、``'high'`` は大きい x。**同点は実際に起きる** — x=6（A=6）では
    Top-K が 0 になり、x=5 との差が浮動小数の下位に落ちる層がある。どちらを
    取ったかで平均 x が動くので、既定を持たせたうえで記録する。
    """
    if prefer not in ('low', 'high'):
        raise ValueError(f"prefer は 'low' か 'high'（{prefer!r}）")
    values, chosen_scores, margins = [], [], []
    for layer, row in enumerate(table):
        items = sorted((int(x), float(v)) for x, v in row.items())
        if not items:
            raise ValueError(f'層 {layer} の候補が空')
        for x, _ in items:
            if not 0 <= x <= n_active:
                raise ValueError(
                    f'層 {layer} の候補 x={x} が 0..{n_active} の外')
        best = min(v for _, v in items)
        ties = [x for x, v in items if v == best]
        pick = min(ties) if prefer == 'low' else max(ties)
        runner_up = min((v for x, v in items if x != pick), default=float('nan'))
        values.append(pick)
        chosen_scores.append(best)
        margins.append(runner_up - best)

    return values, {
        'prefer': prefer,
        'table': [{str(x): float(v) for x, v in row.items()} for row in table],
        'chosen_score': chosen_scores,
        'margin_to_runner_up': margins,
        'sum_local_error': sum(chosen_scores),
        'mean_x': sum(values) / len(values) if values else 0.0,
        'n_routing_layers': sum(1 for x in values if x < n_active),
        'n_no_routing_layers': sum(1 for x in values if x == n_active),
        'degenerate': len(set(values)) <= 1,
    }
