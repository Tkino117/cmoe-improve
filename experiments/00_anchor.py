"""アンカー: 移送した基盤が、既存の測定と同じ数字を出すか。

CMoE-ref の一次結果から2構成を選んで固定値として置いてある。片方はルーターの
経路まで、もう片方は配分の適用まで通る。

  A: 一様 x=3 + 現行ルーター … 軸1〜4 と6
  B: ビーム配分 + 現行ルーター … 上記 + 軸5 の適用側

探索アルゴリズムの再現は求めない（浮動小数の順序ひとつで分岐が変わる）。B は
report/06 の配分ベクトルを定数として適用した結果だけを見る。

  uv run python experiments/00_anchor.py          # 両方
  uv run python experiments/00_anchor.py --only A
"""

import argparse
import json
import os
import sys

from cmoe.cli import main as cli_main

# 出典: report/13 の seed 別 PPL 表（carve n=8, seed 0, N=8, A=6）
ANCHORS = {
    'A': {
        'alloc': 'uniform3',
        'router': 'cmoe',
        'expected': {'wikitext2': 7.071955, 'c4-new': 10.236954},
    },
    'B': {
        'alloc': 'beam',
        'router': 'cmoe',
        'expected': {'wikitext2': 6.941375, 'c4-new': 10.268085},
    },
}

# 同じ構成の seed 違いで PPL は ±0.02〜0.04 動くが、これは同じ seed の再現なので
# 一致するはず。bf16 の非決定性ぶんだけを許容幅にする。
TOLERANCE = 1e-4


def run(name, out_root):
    anchor = ANCHORS[name]
    out = os.path.join(out_root, f'anchor_{name}_{anchor["alloc"]}_{anchor["router"]}')
    cli_main([
        'run',
        '--alloc', anchor['alloc'],
        '--router', anchor['router'],
        '--seeds', '0',
        '--out', out,
    ])
    with open(os.path.join(out, 'summary.json')) as handle:
        summary = json.load(handle)
    measured = {name: row['ppl']
                for name, row in summary['runs'][0]['ppl'].items()}
    rows = []
    for dataset, expected in anchor['expected'].items():
        got = measured[dataset]
        rows.append((dataset, expected, got, abs(got - expected) <= TOLERANCE))
    return rows


def report(name, rows):
    print(f'\nアンカー {name}')
    ok = True
    for dataset, expected, got, matched in rows:
        mark = '一致' if matched else '不一致'
        print(f'  {dataset:<10} 記録 {expected:.6f}  実測 {got:.6f}  '
              f'差 {got - expected:+.6f}  {mark}')
        ok = ok and matched
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=sorted(ANCHORS), default=None)
    parser.add_argument('--out-root', default='result_logs')
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(ANCHORS)
    passed = all(report(name, run(name, args.out_root)) for name in names)
    sys.exit(0 if passed else 1)
