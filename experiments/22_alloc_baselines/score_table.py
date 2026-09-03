"""22-g 校正上の目的関数（`suffix_kl`）を、行 × seed の表にする。GPU は要らない。

**ベンチの前に読む表である。** 提案手法が配分を選ぶのに使っているのがこの値なので、
「探索が最小化しようとしたもの」の上で対照がどこに居るかが、評価指標を1つも見ずに
言える。

`score` は同じ校正トークンの上でしか比べられないので、**seed をまたいで縦に
読んではいけない**（res15 と同じ約束）。列ごとに読む。

  uv run python experiments/22_alloc_baselines/score_table.py --nactive 6
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
LABELS = {
    'uniform': '校正が選んだ一様',
    'reverse': '順序対照: 反転',
    'shuffle1': '順序対照: 並べ替え1',
    'shuffle2': '順序対照: 並べ替え2',
    'lexi': 'LExI 相当',
    'owl_default': 'OWL 相当（既定）',
    'proposal': '**提案（beam 幅4）**',
}
ORDER = ('uniform', 'reverse', 'shuffle1', 'shuffle2', 'lexi', 'owl_default',
         'proposal')


def load(path):
    with (ROOT / path).open() as handle:
        return json.load(handle)


def scores_of(seed, tag, name='score22'):
    record = load(f'result_logs/{name}{tag}_seed{seed}/score.json')
    return {','.join(str(x) for x in row['allocation']['values']): row['score']
            for row in record['scores']}


def spec_for(alloc, name, n_layers):
    if name == 'proposal':
        return alloc['proposal']['spec']
    if name == 'uniform':
        return ','.join([alloc['calibration_uniform'].replace('uniform', '')]
                        * n_layers)
    entry = next((e for e in alloc['candidates']
                  if e['name'] == name or name in e['aliases']), None)
    if entry is None:
        raise SystemExit(f'候補に {name} が無い')
    return entry['spec']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--rows', default=','.join(ORDER))
    args = parser.parse_args(argv)

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    rows = [name.strip() for name in args.rows.split(',') if name.strip()]
    allocs = {s: load(f'result_logs/alloc22{tag}_seed{s}.json') for s in SEEDS}
    scores = {s: scores_of(s, tag) for s in SEEDS}
    n_layers = len(allocs[SEEDS[0]]['proposal']['values'])

    print(f'### 校正上の `suffix_kl`（小さいほど良い。スパース率 '
          f'{100 - 100 * args.nactive // 8}%、N=8 / A={args.nactive}）\n')
    print('`score` は同じ校正トークンの上でしか比べられない。'
          '**seed をまたいで縦に読まないこと。**\n')
    header = '| 配分 | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | 平均 x |'
    print(header)
    print('|---|' + '--:|' * (len(SEEDS) + 1))
    for name in rows:
        cells, mean_x = [], []
        for seed in SEEDS:
            spec = spec_for(allocs[seed], name, n_layers)
            mean_x.append(sum(int(v) for v in spec.split(',')) / n_layers)
            value = scores[seed].get(spec)
            cells.append('—' if value is None else f'{value:.7f}')
        print(f'| {LABELS.get(name, name)} | ' + ' | '.join(cells)
              + f' | {sum(mean_x) / len(mean_x):.3f} |')

    print('\n**提案に対する比**（1 より大きければ提案の方が良い）\n')
    print('| 配分 | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' |')
    print('|---|' + '--:|' * len(SEEDS))
    for name in rows:
        if name == 'proposal':
            continue
        cells = []
        for seed in SEEDS:
            spec = spec_for(allocs[seed], name, n_layers)
            mine = scores[seed].get(spec)
            best = scores[seed].get(allocs[seed]['proposal']['spec'])
            cells.append('—' if mine is None or not best
                         else f'{mine / best:.4f}')
        print(f'| {LABELS.get(name, name)} | ' + ' | '.join(cells) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
