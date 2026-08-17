"""アンカー: 移送した基盤が、既存の測定と同じ数字を出すか。

CMoE-ref の一次結果から5構成を選んで固定値として置いてある。それぞれ通る経路が
違い、合わせて軸1〜6 の全部を覆う。

  A: 一様 x=3 + 現行ルーター      … 軸1〜4 と6
  B: ビーム配分 + 現行ルーター    … 上記 + 軸5 の適用側
  C: 一様 x=3 + 方式4             … 上記 + fit 分割とルーター方式の連鎖
  D: ビーム配分 + 方式4           … 配分とルーターを同時に入れた組み合わせ
  E: 一様 x=3 + |h| オラクル      … 診断用ルーター（配備できない上限）

探索アルゴリズムの再現は求めない（浮動小数の順序ひとつで分岐が変わる）。B・D は
report/06 の配分ベクトルを定数として適用した結果だけを見る。

  uv run python experiments/00_anchor.py          # 全部
  uv run python experiments/00_anchor.py --only C
"""

import argparse
import json
import os
import sys

from cmoe.cli import main as cli_main

# 出典: report/13 の seed 別 PPL 表（carve n=8, fit n=64, seed 0, N=8, A=6）
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
    'C': {
        'alloc': 'uniform3',
        'router': 'oracle_recovery',
        'expected': {'wikitext2': 7.003395, 'c4-new': 10.232112},
    },
    'D': {
        'alloc': 'beam',
        'router': 'oracle_recovery',
        'expected': {'wikitext2': 6.942683, 'c4-new': 10.288317},
    },
    # 出典: report/07 の表（x=3 の行）。この実行は wikitext2 しか測っていない
    'E': {
        'alloc': 'uniform3',
        'router': 'oracle_abs',
        'expected': {'wikitext2': 6.638487},
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
        '--datasets', ','.join(anchor['expected']),
        '--out', out,
    ])
    with open(os.path.join(out, 'summary.json')) as handle:
        summary = json.load(handle)
    measured = {dataset: row['ppl']
                for dataset, row in summary['runs'][0]['ppl'][anchor['router']].items()}
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
    results = [(name, run(name, args.out_root)) for name in names]
    passed = all(report(name, rows) for name, rows in results)
    sys.exit(0 if passed else 1)
