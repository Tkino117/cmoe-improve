"""図: 提案が出した配分の内訳を、25% と 50% で並べる。

1本の棒が1層で、高さは expert 数 N=8 である。積み上げは下から順に

  routed (not activated)  N − A 個。走らない。**この帯の高さがスパース率そのもの**
  routed (activated)      A − x_ℓ 個。トークンごとにルーターが選ぶ
  shared                  x_ℓ 個。全トークンで走る

上段が 25%（A=6）、下段が 50%（A=4）。どちらも slimpajama n=16 seed 0 で校正した
提案（`suffix_kl` + ビーム幅4）の配分である（report/07・09）。

  上段 result_logs/slimpajama_w4_n16_seed0/search.json
  下段 result_logs/slimpajama_a4_w4_n16_seed0/search.json

**2本が同じ校正トークンの上のものかを検査してから描く。** 違うものを並べると
「動作点を変えたときの違い」を見ている図ではなくなる。

matplotlib はこのプロジェクトの依存に入れていない（Docker の sweep イメージに
描画系を入れたくない）。その場で引いて走らせる:

  uv run --with matplotlib python report/figures/fig_allocation_stacked.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'report', 'figures', 'fig_allocation_stacked')

N_EXPERTS = 8
SEED = 0
# 動作点ごとの出どころ。A=6 は無印、A=4 は _a4（report/09 の命名）
SOURCES = {
    6: 'result_logs/slimpajama_w4_n16_seed{seed}/search.json',
    4: 'result_logs/slimpajama_a4_w4_n16_seed{seed}/search.json',
}

# ラベルは英語にする。ICASSP の本文が英語であるのに加え、matplotlib の既定の
# フォントに CJK が無く、日本語を渡すと豆腐になるため
# 図の中の文字は全部この大きさにする（見出しに合わせる。小さい字を混ぜない）
TEXT = 11

SHARED = '#E07B39'        # shared: オレンジ
ACTIVATED = '#F2C14E'     # routed (activated): 黄
INACTIVE = '#4878A8'      # routed (not activated): 青

# 論文体裁は CMoE-ref/figures/fig_alloc_a6_seed0.py に合わせる
# （Times 系セリフ + STIX 数式、フォントは埋め込む）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'Liberation Serif',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def allocations(seed):
    """(A, 配分) を動作点ごとに。校正トークンが揃っていなければ止まる。"""
    records = {n_active: load(path.format(seed=seed))
               for n_active, path in SOURCES.items()}
    hashes = {record['calibration']['token_hash'] for record in records.values()}
    if len(hashes) != 1:
        raise SystemExit(f'校正トークンが揃っていない {sorted(hashes)}')
    out = []
    for n_active, record in records.items():
        if record['arguments']['nactive'] != n_active:
            raise SystemExit(f'A={n_active} の出どころに違う動作点が混ざっている')
        values = list(record['allocation']['values'])
        if max(values) > n_active:
            raise SystemExit(f'A={n_active} を超える x がある')
        out.append((n_active, values))
    return out


def draw(rows, path):
    figure, axeses = plt.subplots(
        len(rows), 1, figsize=(6.5, 3.05),
        gridspec_kw={'hspace': 0.36, 'top': 0.85, 'bottom': 0.16,
                     'left': 0.11, 'right': 0.99})
    bar = dict(width=0.82, linewidth=0.25, edgecolor='white')

    for index, (axes, (n_active, values)) in enumerate(zip(axeses, rows)):
        n_layers = len(values)
        layers = range(n_layers)
        axes.set_axisbelow(True)
        axes.grid(axis='y', color='0.85', linewidth=0.4)

        # 積み上げは下から。走らない分を床に置くと、その帯の高さがスパース率になる
        inactive = [N_EXPERTS - n_active] * n_layers
        routed = [n_active - value for value in values]
        axes.bar(layers, inactive, color=INACTIVE,
                 label=r'routed, not activated ($N-A$)', **bar)
        axes.bar(layers, routed, bottom=inactive, color=ACTIVATED,
                 label=r'routed, activated ($A-x_\ell$)', **bar)
        axes.bar(layers, values, bottom=[a + b for a, b in zip(inactive, routed)],
                 color=SHARED, label=r'shared ($x_\ell$)', **bar)

        axes.set_xlim(-0.7, n_layers - 0.3)
        axes.set_ylim(0, N_EXPERTS)
        axes.set_yticks(range(0, N_EXPERTS + 1, 4))
        axes.set_ylabel('experts / token', fontsize=TEXT, labelpad=3)
        axes.set_xticks(range(0, n_layers, 4))
        axes.set_xticks(layers, minor=True)
        axes.tick_params(axis='both', labelsize=TEXT)
        axes.tick_params(axis='x', which='minor', length=1.5)
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)
        axes.set_title(
            f'{100 * (N_EXPERTS - n_active) // N_EXPERTS}% sparsity '
            f'($N={N_EXPERTS}$, $A={n_active}$)', fontsize=TEXT, pad=4, loc='left')

        # 層の目盛は一番下の1枚だけ。どの段も同じ層を同じ位置に描いている
        if index == len(rows) - 1:
            axes.set_xlabel(r'layer $\ell$', fontsize=TEXT, labelpad=2)
        else:
            axes.set_xticklabels([])

    # 凡例は積み上げの見た目と同じ順（上から shared）にする
    handles, labels = axeses[0].get_legend_handles_labels()
    figure.legend(handles[::-1], labels[::-1], loc='upper center', ncol=3,
                  frameon=False, fontsize=TEXT, handlelength=1.4,
                  handleheight=0.8, handletextpad=0.5, columnspacing=1.8,
                  bbox_to_anchor=(0.5, 1.0))
    for suffix in ('.pdf', '.png'):
        figure.savefig(path + suffix, dpi=600, bbox_inches='tight',
                       pad_inches=0.02)
        print(f'{path + suffix} に書いた')


def main():
    rows = allocations(SEED)
    for n_active, values in rows:
        print(f'A={n_active}（スパース率 '
              f'{100 * (N_EXPERTS - n_active) // N_EXPERTS}%）seed {SEED} '
              f'mean x={sum(values) / len(values):.4f}')
        print('  ' + ','.join(str(value) for value in values))
    draw(rows, OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
