"""22-f ``compare.py --json`` が書いた数値を、そのまま報告に貼れる表にする。

**数字を手で写さないためにある。** 表は3枚:

1. 絶対値（行ごとの平均 x・`acc`・`acc_norm`・`gold_nll`・WikiText-2 PPL）
2. 提案との差（対応再抽出の点推定と 95% 区間）
3. タスクごとの `acc`

  uv run python experiments/22_alloc_baselines/summarize.py \\
      result_logs/compare22_a6.json result_logs/compare22_a4.json
"""

import argparse
import json
import sys

# 表に出す順。compare.py の ROWS と同じ並びで、最後に提案を置く
LABELS = {
    'ew_default': 'EW-rule（既定 τ=0.6, α=[0.2,0.7]）',
    'uniform': '校正が選んだ一様',
    'reverse': '順序対照: 反転',
    'shuffle1': '順序対照: 並べ替え1',
    'shuffle2': '順序対照: 並べ替え2',
    'lexi': 'LExI 相当（層ローカル誤差の argmin）',
    'owl_default': 'OWL 相当（既定 M=5, desc, α=[0.2,0.7]）',
}
DIFF_METRICS = ('acc', 'acc_norm', 'gold_nll')
PPL_NAME = 'wikitext2'


def marked(row, key='mean_difference'):
    star = '*' if row['lower'] > 0 or row['upper'] < 0 else ''
    return (f'{row[key]:+.6f} [{row["lower"]:+.6f}, '
            f'{row["upper"]:+.6f}]{star}')


def absolute_table(payload):
    rows = payload['rows']
    first = next(iter(rows.values()))
    lines = ['| 配分 | 平均 x | `acc` | `acc_norm` | `gold_nll` | WikiText-2 |',
             '|---|--:|--:|--:|--:|--:|']
    for name, row in rows.items():
        mean_x = sum(row['mean_x']) / len(row['mean_x'])
        lines.append(
            f'| {LABELS.get(name, name)} | {mean_x:.3f} | '
            f'{row["base_macro"]["acc"]:.6f} | '
            f'{row["base_macro"]["acc_norm"]:.6f} | '
            f'{row["base_macro"]["gold_nll"]:.4f} | '
            f'{row["ppl"][PPL_NAME]["base_mean_ppl"]:.4f} |')
    cand_x = payload.get('cand_mean_x')
    proposal_x = (f'{sum(cand_x) / len(cand_x):.3f}' if cand_x else '—')
    lines.append(
        f'| **提案（beam 幅4）** | {proposal_x} | '
        f'{first["cand_macro"]["acc"]:.6f} | '
        f'{first["cand_macro"]["acc_norm"]:.6f} | '
        f'{first["cand_macro"]["gold_nll"]:.4f} | '
        f'{first["ppl"][PPL_NAME]["cand_mean_ppl"]:.4f} |')
    return lines


def difference_table(payload):
    rows = payload['rows']
    lines = ['| 対照 | `acc` | `acc_norm` | `gold_nll` | WikiText-2 NLL |',
             '|---|---|---|---|---|']
    for name, row in rows.items():
        cells = [marked(row['difference'][metric]) for metric in DIFF_METRICS]
        cells.append(marked(row['ppl'][PPL_NAME], 'mean_nll_difference'))
        lines.append(f'| {LABELS.get(name, name)} | ' + ' | '.join(cells) + ' |')
    return lines


def task_table(payload):
    rows = payload['rows']
    first = next(iter(rows.values()))
    tasks = list(first['base_tasks'])
    lines = ['| 配分 | ' + ' | '.join(tasks) + ' |',
             '|---|' + '--:|' * len(tasks)]
    for name, row in rows.items():
        cells = [f'{row["base_tasks"][task]["acc"]:.4f}' for task in tasks]
        lines.append(f'| {LABELS.get(name, name)} | ' + ' | '.join(cells) + ' |')
    cells = [f'{first["cand_tasks"][task]["acc"]:.4f}' for task in tasks]
    lines.append('| **提案（beam 幅4）** | ' + ' | '.join(cells) + ' |')
    return lines


def mean_x_note(payload):
    """順序対照の平均 x が提案と一致していることを、表の外に添える数。"""
    rows = payload['rows']
    parts = []
    for name in ('reverse', 'shuffle1', 'shuffle2'):
        if name in rows:
            values = rows[name]['mean_x']
            parts.append(f'{name}={", ".join(f"{v:.4f}" for v in values)}')
    return parts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', help='compare.py --json の出力')
    args = parser.parse_args(argv)

    for path in args.paths:
        with open(path) as handle:
            payload = json.load(handle)
        sparsity = 100 - 100 * payload['n_active'] // 8
        print(f'\n### スパース率 {sparsity}%（N=8 / A={payload["n_active"]}、'
              f'{len(payload["seeds"])} seed 平均）\n')
        print('\n'.join(absolute_table(payload)))
        print('\n**提案 − 対照**（符号は「提案が良い」向きではなく素の差。'
              '`acc` / `acc_norm` は正、`gold_nll` と NLL は負が提案の勝ち。'
              '`*` は95%区間が0をまたがない）\n')
        print('\n'.join(difference_table(payload)))
        print('\n**タスクごと `acc`**\n')
        print('\n'.join(task_table(payload)))
        note = mean_x_note(payload)
        if note:
            print('\n順序対照の seed ごとの平均 x（提案と一致しているはず）: '
                  + ' / '.join(note))
    return 0


if __name__ == '__main__':
    sys.exit(main())
