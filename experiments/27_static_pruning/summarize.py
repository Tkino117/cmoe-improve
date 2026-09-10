"""27 の表を作る。静的プルーニングの対照を提案手法と並べる。GPU は要らない。

出す列は「実効のブロック削減率」を共通軸にする。手法ごとにスパース率の定義が
違い（この基盤の 25% は FFN ニューロンの比でブロック全体の 16.71%、FLAP の
25% は attn+FFN 全体の比、LLM-Pruner の 25% は刈る 26 層の中でのチャネル比）、
名目のスパース率で並べると別々のものを同じ行に置くことになる。

  uv run python experiments/27_static_pruning/summarize.py
  uv run python experiments/27_static_pruning/summarize.py --format markdown
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_runs                                          # noqa: E402
from run import CONTROLS, DENSE_REFERENCE, METHODS, SCOPES, SEEDS, SPARSITIES  # noqa: E402

from cmoe.eval import bench, bench_stats                     # noqa: E402

DATASETS = ('wikitext2', 'c4-new')
METRICS = ('acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement')


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def prune_runs():
    """(手法, scope, スパース率) -> seed ごとの run。"""
    rows = {}
    for method in METHODS:
        for scope in SCOPES:
            path = f'result_logs/prune27_{method}_{scope}'
            if not os.path.exists(os.path.join(ROOT, path, 'summary.json')):
                continue
            payload = load(os.path.join(path, 'summary.json'))
            for run in payload['runs']:
                key = (method, scope, f'{run["sparsity"]:g}')
                rows.setdefault(key, {})[run['seed']] = (path, run)
    return rows


def control_runs(sparsity):
    """提案手法（beam 幅4）の run を seed で引く。"""
    control = CONTROLS[sparsity]
    payloads = {}
    for path in control['dirs']:
        full = os.path.join(ROOT, path, 'summary.json')
        if not os.path.exists(full):
            return None
        payloads[path] = load(full)
    return compare_runs.pick(payloads, control['alloc'], 'cmoe')


def mean_ppl(runs, dataset):
    values = [compare_runs.run_ppl(run, 'cmoe', dataset)['ppl']
              for _, run in runs.values()]
    return sum(values) / len(values)


def mean_bench(runs, reference):
    per_seed = []
    for path, run in runs.values():
        samples = bench.load_samples(os.path.join(
            ROOT, path, compare_runs.run_bench_path(run, 'cmoe')))
        per_seed.append(bench_stats.summarize(samples, reference))
    return {metric: sum(one['macro'][metric] for one in per_seed) / len(per_seed)
            for metric in per_seed[0]['macro']}, per_seed


def accounting_of(runs):
    values = [run['accounting']['effective_block_sparsity']
              for _, run in runs.values()]
    heads = [run['accounting']['mean_heads'] for _, run in runs.values()]
    inter = [run['accounting']['mean_intermediate'] for _, run in runs.values()]
    return (sum(values) / len(values), sum(heads) / len(heads),
            sum(inter) / len(inter))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', default='text', choices=('text', 'markdown'))
    args = parser.parse_args()

    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    dense, _ = {}, None
    dense = bench_stats.summarize(reference)['macro']
    prune = prune_runs()

    for sparsity in SPARSITIES:
        control = control_runs(sparsity)
        rows = []
        if control is not None:
            macro, _ = mean_bench(control, reference)
            moe = next(iter(control.values()))[1]
            rows.append({
                'label': f'提案（beam 幅4）',
                'effective': None, 'heads': None, 'inter': None,
                'ppl': {name: mean_ppl(control, name) for name in DATASETS},
                'macro': macro, 'n_seeds': len(control)})
        for method in METHODS:
            for scope in SCOPES:
                runs = prune.get((method, scope, sparsity))
                if not runs:
                    continue
                effective, heads, inter = accounting_of(runs)
                macro, _ = mean_bench(runs, reference)
                rows.append({
                    'label': f'{method} ({scope})',
                    'effective': effective, 'heads': heads, 'inter': inter,
                    'ppl': {name: mean_ppl(runs, name) for name in DATASETS},
                    'macro': macro, 'n_seeds': len(runs)})
        if not rows:
            continue
        print()
        print(f'== スパース率 {sparsity}（提案手法の数え方。FFN ニューロンの比）==')
        if args.format == 'markdown':
            print('| 手法 | 実効ブロック削減率 | 平均 heads | 平均 inter | '
                  'WikiText-2 | C4 | acc | acc_norm | gold_nll |')
            print('|---|--:|--:|--:|--:|--:|--:|--:|--:|')
            for row in rows:
                effective = ('—' if row['effective'] is None
                             else f'{row["effective"]:.4f}')
                heads = '32.0' if row['heads'] is None else f'{row["heads"]:.1f}'
                inter = ('11008' if row['inter'] is None
                         else f'{row["inter"]:.0f}')
                print(f'| {row["label"]} | {effective} | {heads} | {inter} | '
                      f'{row["ppl"]["wikitext2"]:.4f} | {row["ppl"]["c4-new"]:.4f} | '
                      f'{row["macro"]["acc"]:.4f} | {row["macro"]["acc_norm"]:.4f} | '
                      f'{row["macro"]["gold_nll"]:.4f} |')
        else:
            for row in rows:
                effective = ('     —' if row['effective'] is None
                             else f'{row["effective"]:.4f}')
                print(f'  {row["label"]:<22} 実効={effective} '
                      f'wikitext2={row["ppl"]["wikitext2"]:8.4f} '
                      f'c4={row["ppl"]["c4-new"]:8.4f} '
                      f'acc={row["macro"]["acc"]:.4f} '
                      f'gold_nll={row["macro"]["gold_nll"]:.4f} '
                      f'({row["n_seeds"]} seed)')
    print()
    print(f'dense: acc={dense["acc"]:.4f} acc_norm={dense["acc_norm"]:.4f} '
          f'gold_nll={dense["gold_nll"]:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
