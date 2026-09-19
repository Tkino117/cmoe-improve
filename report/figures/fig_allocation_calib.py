"""図: 校正データの量（n）を変えたとき、層ごとの配分はどう動くか。

Llama-2-7b / スパース率 25%（N=8 / A=6）/ beam 幅2 の提案配分を、
n = 16 / 32 / 64（× 2048 トークン）× seed 5,6,7 の9本で並べる。

  (上) 帯   1行が1本の配分ベクトル。塗りと数字は層 l の shared expert 数 x_l。
            右に平均 x と、その配分の PPL から同じ n の uniform3 を引いた差を添える
  (下) 折れ線 n ごとの seed 平均（3本）。CMoE の uniform3 を破線で置く

`fig_allocation_seeds.py` と同じ塗り（x=A は地の青、削った層が赤で立つ）にして、
2つの図を並べて読めるようにしてある。

出どころは report/31 の測定（`result_logs/`、展開済みのものをそのまま読む）。

  配分  slimpajama_w2_n{n}_seed{s}/search.json
  PPL   ppl31_n{n}_seed{s}/summary.json（uniform3 と探索配分を同じ実行で測ってある）

配分は値の並びで run に突き合わせ、1本に決まらなければ止まる。校正トークンの
ハッシュが9本すべてで相異なることも確かめる（同じデータを2度並べていないか）。

  uv run --with matplotlib python report/figures/fig_allocation_calib.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                            # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, 'result_logs')
OUT = os.path.join(ROOT, 'report', 'figures', 'fig_allocation_calib')

N_ACTIVE = 6      # x は 0..A。A を超える共有は取れないので、色の目盛もここで切る
CMOE_X = 3        # 対照（全層 x = A/2）
NSAMPLES = (16, 32, 64)
SEEDS = (5, 6, 7)
SEARCH = 'slimpajama_w2_n{n}_seed{seed}/search.json'
PPL = 'ppl31_n{n}_seed{seed}/summary.json'
DATASETS = ('wikitext2', 'c4-new')

# n ごとの線の色。校正を増やすほど濃くする（量の順が明るさの順になる）
LINES = ('#f0a38a', '#e0603a', '#8c2d10')

TEXT = 9
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'Liberation Serif',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': TEXT,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def load(path):
    with open(path) as handle:
        return json.load(handle)


def rows():
    """(n, seed, 配分, 平均 x, {データ: 探索 − uniform3}) を n の順に。"""
    out = []
    hashes = {}
    for n_samples in NSAMPLES:
        for seed in SEEDS:
            search = load(os.path.join(
                LOGS, SEARCH.format(n=n_samples, seed=seed)))
            values = list(search['allocation']['values'])
            token_hash = search['calibration']['token_hash']
            if token_hash in hashes:
                raise SystemExit(
                    f'n={n_samples} seed {seed} の校正トークンが '
                    f'{hashes[token_hash]} と同一（{token_hash[:12]}）。'
                    '別の条件として並べてはいけない')
            hashes[token_hash] = f'n={n_samples} seed {seed}'
            if max(values) > N_ACTIVE or min(values) < 0:
                raise SystemExit(
                    f'n={n_samples} seed {seed}: x が 0..{N_ACTIVE} の外 {values}')

            summary = load(os.path.join(
                LOGS, PPL.format(n=n_samples, seed=seed)))
            found = [run for run in summary['runs']
                     if list(run['allocation']['values']) == values]
            baseline = [run for run in summary['runs']
                        if run['allocation']['name'] == f'uniform{CMOE_X}']
            if len(found) != 1 or len(baseline) != 1:
                raise SystemExit(
                    f'n={n_samples} seed {seed}: 探索配分に当たる run が '
                    f'{len(found)} 本、uniform{CMOE_X} が {len(baseline)} 本')
            delta = {name: (found[0]['ppl']['cmoe'][name]['ppl']
                            - baseline[0]['ppl']['cmoe'][name]['ppl'])
                     for name in DATASETS}
            out.append((n_samples, seed, values,
                        search['allocation']['mean_x'], delta))
    lengths = {len(values) for _, _, values, _, _ in out}
    if len(lengths) != 1:
        raise SystemExit(f'層数が揃っていない {sorted(lengths)}')
    return out


def palette():
    """`fig_allocation_seeds.py` と同じ塗り。x=A が地、削った層が赤で立つ。"""
    reds = plt.get_cmap('Reds')(np.linspace(0.72, 0.20, N_ACTIVE))  # x=0..A-1
    colormap = ListedColormap(list(reds) + [plt.get_cmap('Blues')(0.22)])
    norm = BoundaryNorm([level - 0.5 for level in range(N_ACTIVE + 1)]
                        + [N_ACTIVE + 0.5], colormap.N)
    return colormap, norm


def draw(records, path):
    grid = np.array([values for *_, values, _, _ in
                     [(n, s, v, m, d) for n, s, v, m, d in records]])
    n_rows, n_layers = grid.shape
    figure, ((axes, panel), (trend, spare)) = plt.subplots(
        2, 2, figsize=(3.45, 2.75),
        sharex='col',
        gridspec_kw={'width_ratios': [2.15, 0.74],
                     'height_ratios': [1.0, 0.50],
                     'wspace': 0.04, 'hspace': 0.20})
    spare.axis('off')

    colormap, norm = palette()
    axes.imshow(grid, cmap=colormap, norm=norm, aspect='auto')
    for row, line in enumerate(grid):
        for column, value in enumerate(line):
            # 色は一目で見るためのもので、読む値はこの数字である（モノクロ複写でも
            # 残る）。塗りの明るさで字の色を決める
            red, green, blue = colormap(norm(value))[:3]
            dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.55
            axes.text(column, row, str(value), ha='center', va='center',
                      fontsize=3.6, color='white' if dark else 'black')

    # n の切れ目に白い横線を引く。3本ずつの塊が一目で分かるようにする
    for boundary in range(len(SEEDS), n_rows, len(SEEDS)):
        axes.axhline(boundary - 0.5, color='white', linewidth=1.2)

    axes.set_yticks(range(n_rows))
    axes.set_yticklabels([f'$n{{=}}{n}$  s{seed}'
                          for n, seed, *_ in records], fontsize=4.6)
    axes.set_xticks([0, 8, 16, 24, 31])
    axes.tick_params(labelbottom=False)
    axes.tick_params(length=2, pad=1.5)
    for spine in axes.spines.values():
        spine.set_linewidth(0.4)

    panel.axis('off')
    panel.set_xlim(-0.05, 1.05)
    panel.set_ylim(0, 1)
    columns = [
        (0.20, 'Mean\n$x_l$', [f'{mean_x:.2f}' for *_, mean_x, _ in records]),
        (0.62, '$\\Delta$PPL\nWT2',
         [f'{delta["wikitext2"]:+.2f}' for *_, delta in records]),
        (0.97, '$\\Delta$PPL\nC4',
         [f'{delta["c4-new"]:+.2f}' for *_, delta in records]),
    ]
    for offset, header, values in columns:
        panel.text(offset, 1.015, header, fontsize=4.6, ha='center', va='bottom',
                   linespacing=1.1)
        for row, value in enumerate(values):
            # imshow の行は上から並ぶので、上下を合わせるために 1 から引く
            panel.text(offset, 1 - (row + 0.5) / n_rows, value, fontsize=4.6,
                       ha='center', va='center')

    layers = np.arange(n_layers)
    for index, n_samples in enumerate(NSAMPLES):
        block = np.array([values for n, _, values, _, _ in records
                          if n == n_samples])
        trend.plot(layers, block.mean(axis=0), color=LINES[index],
                   linewidth=0.9, label=f'$n{{=}}{n_samples}$')
    trend.axhline(CMOE_X, color='#2a78d6', linewidth=0.7, linestyle='--',
                  label=f'CMoE ($x_l{{=}}{CMOE_X}$)')
    trend.set_xlim(-0.5, n_layers - 0.5)
    trend.set_ylim(-0.2, N_ACTIVE + 0.2)
    trend.set_xticks([0, 8, 16, 24, 31])
    trend.set_xticklabels(['0', '8', '16', '24', '31'], fontsize=5.5)
    trend.set_yticks(range(0, N_ACTIVE + 1, 2))
    trend.set_yticklabels([str(value) for value in range(0, N_ACTIVE + 1, 2)],
                          fontsize=5.5)
    trend.set_xlabel('Layer $l$', fontsize=5.5, labelpad=1)
    trend.set_ylabel('$x_l$', fontsize=5.5, labelpad=2)
    trend.tick_params(length=2, pad=1.5)
    # 凡例は軸の外に出す。中に置くと3本の折れ線に重なる
    trend.legend(fontsize=4.6, frameon=False, loc='upper center',
                 ncol=4, handlelength=1.4, borderaxespad=0,
                 columnspacing=1.0, bbox_to_anchor=(0.5, -0.52))
    for spine in ('top', 'right'):
        trend.spines[spine].set_visible(False)

    # 帯と折れ線は x を共有させてある。tight_layout は軸を動かして揃いを崩すので
    # 使わない（`axis('off')` の軸が混ざると警告も出る）
    figure.subplots_adjust(left=0.16, right=0.99, top=0.92, bottom=0.14)
    for suffix in ('.pdf', '.png'):
        figure.savefig(f'{path}{suffix}', dpi=400, bbox_inches='tight')
        print(f'{path}{suffix} に書いた')


def main():
    records = rows()
    draw(records, OUT)

    print()
    for n_samples in NSAMPLES:
        block = np.array([values for n, _, values, _, _ in records
                          if n == n_samples])
        spread = block.max(axis=0) - block.min(axis=0)
        deltas = [delta for n, _, _, _, delta in records if n == n_samples]
        print(f'n={n_samples}: 平均 x {block.mean():.3f}、'
              f'層ごとの seed 標準偏差の平均 {block.std(axis=0, ddof=1).mean():.3f}、'
              f'3 seed 一致 {int((spread == 0).sum())}/{block.shape[1]}')
        for name in DATASETS:
            values = [delta[name] for delta in deltas]
            print(f'  {name:9s} ΔPPL 平均 {np.mean(values):+.4f}  '
                  f'({", ".join(f"{v:+.4f}" for v in values)})')


if __name__ == '__main__':
    main()
