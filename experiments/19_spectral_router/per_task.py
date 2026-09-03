"""本測定の生の尤度から、タスクごとの対照比を出す。GPU は使わない。

`run.py` が書く `summary.json` の `bench_summary` はマクロ平均の区間しか持たない
（タスク別は点推定だけ）。ここは `bench/*.json` に残っている問題ごとの対数尤度を
読み直して、**タスクごとに**対照 `cmoe` との差を再抽出する。層は seed で、単位は
問題である — report/18 の `compare_vs_cmoe_per_task.md` と同じ取り方にしてある。

    uv run python experiments/19_spectral_router/per_task.py \
        --run result_logs/spectral_router_bench
"""

import argparse
import json
import os

from cmoe.eval import bench, bench_stats
from cmoe.eval.stats import stratified_paired_bootstrap

METRICS = ('acc', 'acc_norm', 'gold_nll', 'margin', 'ref_agreement', 'ref_kl')
REPS, BOOT_SEED = 10000, 20260813


def load_run(path, baseline):
    """(ルーター名の並び, ルーター -> seed -> タスク -> 問題ごとの記録, dense)。"""
    with open(os.path.join(path, 'summary.json')) as handle:
        payload = json.load(handle)
    routers = list(payload['runs'][0]['bench_samples'])
    if baseline not in routers:
        raise SystemExit(f'{path} に対照 {baseline} が無い')
    routers = [baseline] + [name for name in routers if name != baseline]
    samples = {name: {} for name in routers}
    for run in payload['runs']:
        for name in routers:
            samples[name][run['seed']] = bench.load_samples(
                os.path.join(path, run['bench_samples'][name]))
    reference = bench.load_samples(os.path.join(path, 'bench', 'dense.json'))
    return routers, samples, reference


def averages(samples, reference, seeds):
    """seed 間の平均（タスク × 指標）。"""
    per_seed = [bench_stats.summarize(samples[seed], reference) for seed in seeds]
    return {task: {metric: sum(one['tasks'][task][metric] for one in per_seed)
                   / len(per_seed) for metric in METRICS}
            for task in per_seed[0]['tasks']}


def compare(base, cand, reference, seeds, tasks):
    """タスクごとの対照比。seed を層とした問題単位の対応再抽出。"""
    rows = {}
    for task in tasks:
        rows[task] = {}
        for metric in METRICS:
            strata = [bench_stats.paired_differences(
                base[seed][task], cand[seed][task], metric,
                reference=reference.get(task)) for seed in seeds]
            rows[task][metric] = stratified_paired_bootstrap(
                strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
                unit='paired-question-within-seed',
                lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])
    return rows


def interval(entry, digits):
    mark = '*' if entry['lower'] > 0 or entry['upper'] < 0 else ''
    text = (f'{entry["mean_difference"]:+.{digits}f} '
            f'[{entry["lower"]:+.{digits}f}, {entry["upper"]:+.{digits}f}]')
    if mark:
        text = f'**{text}***'
    return f'{text} {entry["improved_seeds"]}/{entry["n_seeds"]}'


def render(routers, means, dense, paired, tasks, metrics):
    lines = []
    for metric in metrics:
        digits = 4
        lines += ['', f'**`{metric}`**', '',
                  '| タスク | dense | ' + ' | '.join(f'`{r}`' for r in routers)
                  + ' | ' + ' | '.join(f'`{r}` 対照比' for r in routers[1:]) + ' |',
                  '|---' * (2 + len(routers) + len(routers) - 1) + '|']
        for task in tasks:
            cells = [task, f'{dense[task][metric]:.4f}']
            cells += [f'{means[r][task][metric]:.4f}' for r in routers]
            cells += [interval(paired[r][task][metric], digits)
                      for r in routers[1:]]
            lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='result_logs/spectral_router_bench')
    parser.add_argument('--baseline', default='cmoe')
    parser.add_argument('--metrics', default='acc,gold_nll',
                        help=f'出す指標。既定以外は {",".join(METRICS)} から選ぶ')
    parser.add_argument('--out', default=None, help='書き出す markdown')
    args = parser.parse_args(argv)

    routers, samples, reference = load_run(args.run, args.baseline)
    seeds = sorted(samples[args.baseline])
    tasks = list(samples[args.baseline][seeds[0]])
    means = {name: averages(samples[name], reference, seeds) for name in routers}
    dense = bench_stats.summarize(reference, reference)['tasks']
    paired = {name: compare(samples[args.baseline], samples[name], reference,
                            seeds, tasks) for name in routers[1:]}

    metrics = [name for name in args.metrics.split(',') if name.strip()]
    lines = [f'# {args.run} タスクごと（seed {seeds}、対照 `{args.baseline}`）', '',
             '対照比は seed を層とした問題単位の対応再抽出'
             f'（{REPS}回 / seed {BOOT_SEED}）、`*` は95%区間が0をまたがないもの。']
    lines += render(routers, means, dense, paired, tasks, metrics)
    text = '\n'.join(lines) + '\n'
    if args.out:
        with open(args.out, 'w') as handle:
            handle.write(text)
        print(f'{args.out} に書いた')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
