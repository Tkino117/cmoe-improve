"""図1: 層単独で決めた配分は、全層共通の配分とほとんど変わらない。

観察2（層間の依存）の図。**3本の配分ベクトルを色帯で並べ、右に PPL を添える**
だけである。平均 x は図に載せず、標準出力にだけ出す（本文で使うときはそこから引く）。

  (a) 全層共通の最良配分   校正が `suffix_kl` で選んだ一様（report/07）
  (b) 層単独の独立 argmin  層ローカル誤差で層ごとに独立に決めた配分
                           （report/24 = report/22 の LExI 相当）
  (c) 提案                 後続まで通した誤差 + ビーム幅4（report/07）

**(a) と (b) がほぼ同じ塗りで、PPL も同じところに留まる。(c) だけが模様を持ち、
PPL が下がる。** 本文に数字を並べずに済ませるための図なので、本文に残す数字は
「約3割」1つだけにできる。

**帯は seed 0 の配分、PPL は 3 seed の平均である。** 行は固定のベクトルではなく
「配分の決め方」なので、seed ごとに出る配分が違う（校正が選んだ一様は seed 1 だけ
`uniform5`）。キャプションにそう書くこと。

**検査を2つ通してから描く。**

1. 3本の配分が同じ校正トークンの上のものか（違えば「決め方の違い」の図ではない）
2. (b) は別の run で測ってあるので、両方の run に入っている `uniform3` の
   mean NLL が一致するか（report/22 と同じ手続き。合わなければ run をまたいで
   PPL を並べてはいけない）

matplotlib はこのプロジェクトの依存に入れていない（Docker の sweep イメージに
描画系を入れたくない）。その場で引いて走らせる:

  uv run --with matplotlib python report/figures/fig_allocation_bands.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                            # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'report', 'figures', 'fig_allocation_bands')

N_EXPERTS = 8
N_ACTIVE = 6      # x は 0..A。A を超える共有は取れないので、色の目盛もここで切る
SEEDS = (0, 1, 2)
BAND_SEED = 0     # 帯に描く配分の seed
DATASETS = ('wikitext2', 'c4-new')

# 配分の出どころ（seed ごと）
UNIFORM_SCORE = 'result_logs/score_uniform_slimpajama_n16_seed{seed}/score.json'
LOCAL = 'result_logs/lexi_probe_slimpajama_seed{seed}/lexi.json'
LOCAL_SEED0 = 'result_logs/order_local_error_reverse_slim_n16_seed0/search.json'
PROPOSED = 'result_logs/slimpajama_w4_n16_seed{seed}/search.json'
# PPL の出どころ。(a)(c) は report/06・07、(b) は report/22 の run
BENCH = 'result_logs/bench_slimpajama_seed{seed}'
BENCH_LOCAL = 'result_logs/bench22_seed{seed}'
RECHECK_ALLOC = 'uniform3'      # 2つの run に共通で入っている配分

# ラベルは英語にする。ICASSP の本文が英語であるのに加え、matplotlib の既定の
# フォントに CJK が無く、日本語を渡すと豆腐になるため
# 行のラベルは1語にする。図の幅の大半は帯に回したいので、決め方の説明
# （best fixed / independent / beam width 4）はキャプションに置く
LABELS = ('Uniform', 'Layer-local', 'Proposed')


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def allocations(seed):
    """その seed の3本の配分。校正トークンが揃っていなければ止まる。"""
    uniform = load(UNIFORM_SCORE.format(seed=seed))
    local = load(LOCAL.format(seed=seed))
    proposed = load(PROPOSED.format(seed=seed))
    hashes = {record['calibration']['token_hash']
              for record in (uniform, local, proposed)}
    if len(hashes) != 1:
        raise SystemExit(f'seed {seed}: 校正トークンが揃っていない {sorted(hashes)}')

    best = uniform['best']['name']
    if not best.startswith('uniform'):
        raise SystemExit(f'seed {seed}: 校正が選んだ配分が一様ではない {best}')
    n_layers = len(proposed['allocation']['values'])
    rows = [[int(best[len('uniform'):])] * n_layers,
            list(local['allocation']['values']),
            list(proposed['allocation']['values'])]
    if any(len(row) != n_layers for row in rows):
        raise SystemExit(f'seed {seed}: 層数が揃っていない')
    if seed == 0:
        # report/24 の 2026-08-25 の測定と、report/22 の LExI 相当は同じものである
        # （報告書がそう書いている）。図がどちらを描いているか曖昧にしない
        legacy = load(LOCAL_SEED0)['allocation']['values']
        if list(legacy) != rows[1]:
            raise SystemExit('seed 0 の層単独の配分が report/24 と一致しない')
    return rows


def pick(directory, wanted):
    """ベンチ実行から run を1本引く。一様は名前で、探索の結果はベクトルで引く。"""
    payload = load(os.path.join(directory, 'summary.json'))
    found = [run for run in payload['runs']
             if (run['allocation']['name'] == wanted if isinstance(wanted, str)
                 else list(run['allocation']['values']) == list(wanted))]
    if len(found) != 1:
        raise SystemExit(f'{directory} に配分 {wanted} の run が {len(found)} 本ある')
    return found[0]


def recheck():
    """(b) の run と (a)(c) の run が突き合わせられるか。差が0 でなければ止まる。"""
    for seed in SEEDS:
        left = pick(BENCH.format(seed=seed), RECHECK_ALLOC)
        right = pick(BENCH_LOCAL.format(seed=seed), RECHECK_ALLOC)
        for name in DATASETS:
            gap = (right['ppl']['cmoe'][name]['mean_nll']
                   - left['ppl']['cmoe'][name]['mean_nll'])
            if gap:
                raise SystemExit(
                    f'seed {seed} の {RECHECK_ALLOC} が run をまたいで一致しない'
                    f'（{name} Δ={gap:+.3e}）。PPL を並べてはいけない')
    print(f'再現の検査: {RECHECK_ALLOC} は {len(SEEDS)} seed とも Δ=0')


def perplexities(rows_by_seed):
    """行ごとの PPL（seed 平均）。(b) だけ別の run から引く。"""
    out = []
    for index in range(3):
        directory = BENCH_LOCAL if index == 1 else BENCH
        means = {name: [] for name in DATASETS}
        for seed in SEEDS:
            run = pick(directory.format(seed=seed), rows_by_seed[seed][index])
            for name in DATASETS:
                means[name].append(run['ppl']['cmoe'][name]['ppl'])
        out.append({name: sum(values) / len(values)
                    for name, values in means.items()})
    return out


def draw(grid, ppl, path):
    """左に色帯、右に数値の列。列は別の軸に置く（帯の軸に食い込ませない）。"""
    figure, (axes, panel) = plt.subplots(
        1, 2, figsize=(3.45, 1.25),
        gridspec_kw={'width_ratios': [2.15, 0.8], 'wspace': 0.04})
    n_rows = len(grid)

    levels = range(N_ACTIVE + 1)       # x が取りうるのは 0..A
    # x=6（共有を削っていない層）は紙面の地になるので、薄い青1色に置く。削った層
    # だけが赤で立ち、深いほど濃くなる。連続な colormap を 0..6 に張ると x=6 が
    # 濃くなって地が重くなるので、6 とそれ以外で分けて色を作る
    reds = plt.get_cmap('Reds')(np.linspace(0.72, 0.20, N_ACTIVE))  # x=0..5
    colormap = ListedColormap(list(reds) + [plt.get_cmap('Blues')(0.22)])
    norm = BoundaryNorm([level - 0.5 for level in levels] + [levels[-1] + 0.5],
                        colormap.N)
    axes.imshow(grid, cmap=colormap, norm=norm, aspect='auto')

    for row, line in enumerate(grid):
        for column, value in enumerate(line):
            # 色は一目で見るためのもので、読む値はこの数字である（モノクロ複写でも
            # 残る）。カラーバーを置かずに済むので、その幅を数値の列に回せる
            # 塗りの明るさで字の色を決める（淡い帯の上の白抜きは読めない）
            red, green, blue = colormap(norm(value))[:3]
            dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.55
            axes.text(column, row, str(value), ha='center', va='center',
                      fontsize=3.6, color='white' if dark else 'black')

    axes.set_yticks(range(n_rows))
    axes.set_yticklabels(LABELS, fontsize=5.5)
    axes.set_xticks([0, 8, 16, 24, 31])
    axes.set_xticklabels(['0', '8', '16', '24', '31'], fontsize=5.5)
    axes.set_xlabel('Layer', fontsize=5.5, labelpad=1)
    axes.tick_params(length=2, pad=1.5)
    for spine in axes.spines.values():
        spine.set_linewidth(0.4)

    panel.axis('off')
    panel.set_xlim(-0.05, 1.05)
    panel.set_ylim(0, 1)
    # 何の数字かはヘッダで言い切る。1行に収まる幅がないので PPL を下に折る
    columns = [(0.26, 'WikiText-2\nPPL', [f'{row["wikitext2"]:.2f}' for row in ppl]),
               (0.80, 'C4\nPPL', [f'{row["c4-new"]:.2f}' for row in ppl])]
    for offset, header, values in columns:
        panel.text(offset, 1.015, header, fontsize=5, ha='center', va='bottom',
                   linespacing=1.1)
        for row, value in enumerate(values):
            # imshow の行は上から並ぶので、上下を合わせるために 1 から引く
            panel.text(offset, 1 - (row + 0.5) / n_rows, value, fontsize=5.5,
                       ha='center', va='center')

    figure.tight_layout(pad=0.3)
    for suffix in ('.pdf', '.png'):
        figure.savefig(path + suffix, dpi=400, bbox_inches='tight')
        print(f'{path + suffix} に書いた')


def main():
    recheck()
    rows_by_seed = {seed: allocations(seed) for seed in SEEDS}
    grid = rows_by_seed[BAND_SEED]
    means_x = [sum(row) / len(row) for row in grid]
    ppl = perplexities(rows_by_seed)

    for label, row, mean_x, values in zip(LABELS, grid, means_x, ppl):
        print(f'{label:<12} mean x={mean_x:.4f}  '
              + '  '.join(f'{name} {values[name]:.4f}' for name in DATASETS))
        print('    ' + ','.join(str(value) for value in row))
    same = sum(1 for a, b in zip(grid[0], grid[1]) if a == b)
    differ = sum(1 for a, b in zip(grid[1], grid[2]) if a != b)
    print(f'\nseed {BAND_SEED}: (a) と (b) が一致する層 {same}/{len(grid[0])}、'
          f'(b) と (c) が違う層 {differ}/{len(grid[0])}')
    draw(grid, ppl, OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
