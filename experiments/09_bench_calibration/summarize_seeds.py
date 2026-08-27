"""09 の複数 seed をまとめる。GPU は使わない。

各 seed の ``bench_benchtrain_seed<N>/summary.json`` と ``bench/*.json`` を読み、
構成ごとに seed 間の平均と、対照（``uniform4``）との対応のある差を出す。生の
対数尤度が残っているので、指標も比較もここで作り直せる（測り直しは要らない）。

  uv run python experiments/09_bench_calibration/summarize_seeds.py --seeds 0,1,2

**ここが答えるのは「この校正の中での配分の効き」だけである。** 主題である
「校正を替えて何か動いたか」は校正どうしの比較なので、``experiments/compare_runs.py``
が引き受ける（report/04 の wikitext2 と seed で対応づける）。

配分は seed ごとに変わるので、突き合わせる鍵はベクトルではなく「同じ手続きで
作った配分」である。一様は名前で、探索の結果は**幅**で引く。

区間は seed を層とした対応のあるブートストラップで、PPL は評価塊、ベンチは
(seed × タスク) を層とした問題単位である。seed 自体は再抽出しない。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import BASELINE, CALIB, NSAMPLES, ROOT, UNIFORMS, WIDTHS  # noqa: E402

from cmoe.eval import bench, bench_stats  # noqa: E402
from cmoe.eval.stats import (paired_differences,  # noqa: E402
                             stratified_paired_bootstrap)

METRICS = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
DATASETS = ['wikitext2', 'c4-new']
REPS, BOOT_SEED = 10000, 20260813
ROUTER = 'cmoe'


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def searched_labels(root, seed, nsamples, widths):
    """探索が出したベクトル -> ラベル。幅が違っても同じ配分に着くことがある。"""
    labels = {}
    for width in widths:
        path = root / f'{CALIB}_w{width}_n{nsamples}_seed{seed}' / 'search.json'
        if not path.exists():
            continue
        spec = ','.join(str(value) for value in load(path)['allocation']['values'])
        labels[spec] = (f'{labels[spec]}/{width}' if spec in labels
                        else f'beam 幅{width}')
    return labels


def load_seed(root, seed, nsamples, widths):
    """1 seed 分。ラベル -> (run, ベンチの生の尤度)。"""
    out_dir = root / f'bench_{CALIB}_seed{seed}'
    payload = load(out_dir / 'summary.json')
    labels = searched_labels(root, seed, nsamples, widths)
    rows = {}
    for run in payload['runs']:
        name = run['allocation']['name']
        spec = ','.join(str(value) for value in run['allocation']['values'])
        label = labels.get(spec, name)
        samples = bench.load_samples(
            os.path.join(out_dir, run['bench_samples'][ROUTER]))
        rows[label] = (run, samples)
    reference = bench.load_samples(out_dir / 'bench' / 'dense.json')
    return rows, reference


def bench_means(rows, reference):
    """ラベル -> 指標 -> seed 平均のマクロ値。"""
    out = {}
    for label in rows[0]:
        per_seed = [bench_stats.summarize(seed_rows[label][1], reference)
                    for seed_rows in rows]
        out[label] = {metric: sum(one['macro'][metric] for one in per_seed)
                      / len(per_seed) for metric in METRICS}
    return out


def bench_paired(rows, reference, baseline):
    """対照との差。層は (seed × タスク)。"""
    out = {}
    for label in rows[0]:
        if label == baseline:
            continue
        out[label] = {}
        for metric in METRICS:
            strata = []
            for seed_rows in rows:
                left = seed_rows[baseline][1]
                right = seed_rows[label][1]
                for task, samples in right.items():
                    strata.append(bench_stats.paired_differences(
                        left[task], samples, metric,
                        reference=reference.get(task)))
            result = stratified_paired_bootstrap(
                strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
                unit='paired-question-within-seed-and-task',
                lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])
            out[label][metric] = result
    return out


def ppl_table(rows, baseline):
    """ラベル -> データセット -> (seed 平均 PPL, 対照との差)。"""
    out = {}
    for label in rows[0]:
        out[label] = {}
        for name in DATASETS:
            values, strata = [], []
            for seed_rows in rows:
                run = seed_rows[label][0]
                values.append(run['ppl'][ROUTER][name]['ppl'])
                if label != baseline:
                    left = seed_rows[baseline][0]['ppl'][ROUTER][name]
                    strata.append(paired_differences(
                        SimpleNamespace(chunk_mean_nlls=left['chunk_mean_nlls']),
                        SimpleNamespace(
                            chunk_mean_nlls=run['ppl'][ROUTER][name]['chunk_mean_nlls'])))
            entry = {'mean_ppl': sum(values) / len(values)}
            if strata:
                entry['paired'] = stratified_paired_bootstrap(
                    strata, reps=REPS, seed=BOOT_SEED)
            out[label][name] = entry
    return out


def interval(result, key):
    low, high = result['lower'], result['upper']
    mark = '*' if low > 0 or high < 0 else ' '
    return (f'{result[key]:+.6f} [{low:+.6f}, {high:+.6f}]{mark} '
            f'{result["improved_seeds"]}/{result["n_seeds"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--nsamples', type=int, default=NSAMPLES)
    parser.add_argument('--widths', default=','.join(str(w) for w in WIDTHS))
    parser.add_argument('--root', default=str(ROOT / 'result_logs'))
    args = parser.parse_args(argv)

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    widths = [int(field) for field in args.widths.split(',') if field.strip()]
    root = Path(args.root)

    loaded, reference = [], None
    for seed in seeds:
        rows, reference = load_seed(root, seed, args.nsamples, widths)
        loaded.append(rows)
    # seed をまたいでラベルが揃っていることが、対応のある比較の前提である
    common = set(loaded[0])
    for rows in loaded[1:]:
        common &= set(rows)
    missing = [label for rows in loaded for label in rows if label not in common]
    if missing:
        raise SystemExit(f'seed をまたいで揃っていない構成がある: {sorted(set(missing))}')
    order = [label for label in UNIFORMS if label in common]
    order += [label for label in loaded[0] if label not in order]

    print(f'校正 {CALIB} n={args.nsamples} / seed {seeds} / 対照 {BASELINE}')

    means = bench_means(loaded, reference)
    print(f'\n== ベンチ マクロ平均（{len(seeds)} seed 平均）==')
    print('  ' + f'{"":<14}' + ''.join(f'{metric:>14}' for metric in METRICS))
    for label in order:
        print(f'  {label:<14}'
              + ''.join(f'{means[label][metric]:>14.6f}' for metric in METRICS))

    paired = bench_paired(loaded, reference, BASELINE)
    print(f'\n== 対照 {BASELINE} との差（* は95%区間が0をまたがない）==')
    for label in order:
        if label == BASELINE:
            continue
        print(f'  {label}')
        for metric in METRICS:
            print(f'    {metric:<16}{interval(paired[label][metric], "mean_difference")}')

    ppl = ppl_table(loaded, BASELINE)
    print(f'\n== PPL（{len(seeds)} seed 平均。差は共通評価塊の NLL）==')
    for label in order:
        row = ' '.join(f'{name} {ppl[label][name]["mean_ppl"]:.6f}'
                       for name in DATASETS)
        print(f'  {label:<14}{row}')
        for name in DATASETS:
            entry = ppl[label][name]
            if 'paired' in entry:
                print(f'    {name:<16}'
                      + interval(entry['paired'], 'mean_nll_difference'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
