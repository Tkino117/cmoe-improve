"""08 の複数 seed をまとめる。GPU は使わない。

各 seed の ``bench_wikitext2_e16a12_seed<N>/`` を読み、役割ごとに seed 間の平均と
対応のある差を出す。生の対数尤度が残っているので、指標も比較もここで作り直せる。

  uv run python experiments/08_granularity16/summarize_seeds.py --seeds 0,1,2

配分は seed ごとに変わるので、突き合わせる鍵はベクトルではなく**役割**である。
``bench_stage`` が並べる順がそのまま役割になる:

  0. ``uniform6`` … 固定対照。選び方に評価指標が入らない
  1. 校正で選んだ一様 … 一様13本を探索と同じオラクルで採点した最良。探索も対照も
     「校正だけを見て選んだ1本」になるので、これが対等な比較である
  2. 探索配分（beam 幅2）

区間はすべて **seed を層とした対応のあるブートストラップ**で、PPL は評価塊、
ベンチは (seed × タスク) を層とした問題単位である。seed 自体は再抽出しない。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from cmoe.eval import bench, bench_stats
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
ROLES = ['固定対照 uniform6', '校正が選んだ一様', '探索 beam 幅2']
METRICS = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
DATASETS = ['wikitext2', 'c4-new']
REPS, BOOT_SEED = 10000, 20260813
ROUTER = 'cmoe'


def load_seed(seed):
    out = ROOT / 'result_logs' / f'bench_wikitext2_e16a12_seed{seed}'
    with (out / 'summary.json').open() as handle:
        payload = json.load(handle)
    if len(payload['runs']) != len(ROLES):
        raise SystemExit(f'seed {seed} の構成が {len(payload["runs"])} 本')
    return out, payload


def show(label, paired, key):
    low, high = paired['lower'], paired['upper']
    mark = '*' if low > 0 or high < 0 else ' '
    strata = paired.get('improved_strata', paired.get('improved_seeds'))
    total = paired.get('n_strata', paired.get('n_seeds'))
    print(f'    {label:<16} {paired[key]:+.6f} '
          f'[{low:+.6f}, {high:+.6f}]{mark} {strata}/{total}')


def ppl_paired(runs, base, cand, name):
    strata = [paired_differences(
        SimpleNamespace(chunk_mean_nlls=row[base]['ppl'][ROUTER][name]['chunk_mean_nlls']),
        SimpleNamespace(chunk_mean_nlls=row[cand]['ppl'][ROUTER][name]['chunk_mean_nlls']))
        for row in runs]
    return stratified_paired_bootstrap(strata, reps=REPS, seed=BOOT_SEED)


def bench_paired(samples, base, cand, metric, reference):
    strata = []
    for row in samples:
        for task, mine in row[cand].items():
            strata.append(bench_stats.paired_differences(
                row[base][task], mine, metric,
                reference=reference.get(task) if reference else None))
    result = stratified_paired_bootstrap(
        strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
        unit='paired-question-within-seed-and-task',
        lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])
    result['improved_strata'] = result.pop('improved_seeds')
    result['n_strata'] = result.pop('n_seeds')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    args = parser.parse_args(argv)
    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]

    runs, samples, vectors, chosen = [], [], [], []
    reference = None
    for seed in seeds:
        out, payload = load_seed(seed)
        runs.append(payload['runs'])
        samples.append([bench.load_samples(out / row['bench_samples'][ROUTER])
                        for row in payload['runs']])
        vectors.append(payload['runs'][2]['allocation']['values'])
        chosen.append(payload['runs'][1]['allocation']['name'])
        if reference is None:
            reference = bench.load_samples(out / 'bench' / 'dense.json')

    print(f'seed {seeds} / N=16 A=12（スパース率25%）/ 校正 wikitext2 n=8 / '
          f'オラクル local_error / 探索 beam 幅2')
    print(f'校正が選んだ一様: {", ".join(chosen)}')
    for seed, values in zip(seeds, vectors):
        mean_x = sum(values) / len(values)
        print(f'  探索配分 seed {seed} (平均 x={mean_x:.3f}): '
              f'{",".join(str(v) for v in values)}')

    print('\n== PPL（3 seed 平均）==')
    print(f'  {"":<20}' + ''.join(f'{name:>14}' for name in DATASETS))
    for index, role in enumerate(ROLES):
        cells = ''.join(
            f'{sum(row[index]["ppl"][ROUTER][name]["ppl"] for row in runs) / len(runs):>14.6f}'
            for name in DATASETS)
        print(f'  {role:<20}{cells}')

    print('\n== ベンチ マクロ平均（3 seed 平均）==')
    print(f'  {"":<20}' + ''.join(f'{m:>14}' for m in METRICS))
    macro = []
    for index, role in enumerate(ROLES):
        per_seed = [bench_stats.summarize(row[index], reference) for row in samples]
        row = {m: sum(one['macro'][m] for one in per_seed) / len(per_seed)
               for m in METRICS}
        macro.append(row)
        print(f'  {role:<20}' + ''.join(f'{row[m]:>14.6f}' for m in METRICS))

    for base, cand, title in ((0, 2, '探索 − 固定対照 uniform6'),
                              (1, 2, '探索 − 校正が選んだ一様'),
                              (0, 1, '校正が選んだ一様 − 固定対照 uniform6')):
        print(f'\n== {title}（* は95%区間が0をまたがない）==')
        for name in DATASETS:
            show(f'PPL {name}', ppl_paired(runs, base, cand, name),
                 'mean_nll_difference')
        for metric in METRICS:
            show(metric, bench_paired(samples, base, cand, metric, reference),
                 'mean_difference')

    print('\n== タスクごと（acc / acc_norm / gold_nll、3 seed 平均）==')
    tasks = list(samples[0][0])
    for index, role in enumerate(ROLES):
        per_seed = [bench_stats.summarize(row[index], reference) for row in samples]
        print(f'  {role}')
        for task in tasks:
            cell = {m: sum(one['tasks'][task][m] for one in per_seed) / len(per_seed)
                    for m in ('acc', 'acc_norm', 'gold_nll')}
            print(f'    {task:<16} {cell["acc"]:.4f} / {cell["acc_norm"]:.4f} / '
                  f'{cell["gold_nll"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
