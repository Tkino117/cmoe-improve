"""12 の表を作る。行が shared 数 x（S3〜S6）、列が評価タスク。GPU は要らない。

**対角しか無い表である。** どのマスも「そのタスクの train split で校正し、
そのタスクで測った」もので、report/12 の対角を x 方向へ伸ばしたものにあたる。

x=6 は Top-K=0、つまりルーティングが消えた静的マスク（枝刈りの退化ケース）で
ある。この列の中で x=6 が最良なら、この動作点で MoE を選ぶ理由は無い。

uniform4 の行は report/12 の測定をそのまま読む（`bench_calibtask_*`）。
ARC は report/12 と同じく ARC-e と ARC-c を束ねた1タスクとして扱う。

  uv run python experiments/12_shared_ratio_diagonal/summarize.py --seeds 0,1,2
"""

import argparse
import json
import os

from cmoe.eval import bench, bench_stats
from cmoe.eval.stats import stratified_paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPS, BOOT_SEED = 10000, 20260813
ROUTER = 'cmoe'
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

ARC = ('arc_easy', 'arc_challenge')
TASKS = ('piqa', 'winogrande', 'arc', 'hellaswag')
SHORT = {'piqa': 'PIQA', 'winogrande': 'Wino', 'arc': 'ARC',
         'hellaswag': 'HSwag'}
# 行の並び。uniform4 だけ report/12 の出力を読む
ALLOCS = ('uniform3', 'uniform4', 'uniform5', 'uniform6')
PATTERNS = {'uniform4': 'result_logs/bench_calibtask_%s_seed{seed}'}
DEFAULT_PATTERN = 'result_logs/bench_shared_%s_{alloc}_seed{seed}'
# 表の左端。x と Top-K を並べる
LABEL = {'uniform3': 'S3A3E8', 'uniform4': 'S4A2E8',
         'uniform5': 'S5A1E8', 'uniform6': 'S6A0E8'}
BASELINE = 'uniform6'


def merge_arc(samples):
    """ARC-e と ARC-c を、問題を連結した1タスクにする（report/12 と同じ順）。"""
    if not all(name in samples for name in ARC):
        return samples
    parts = [samples[name] for name in ARC]
    merged = bench.TaskSamples(
        task='arc',
        gold=[value for part in parts for value in part.gold],
        loglikelihoods=[row for part in parts for row in part.loglikelihoods],
        choice_lengths=[row for part in parts for row in part.choice_lengths],
        doc_hashes=[value for part in parts for value in part.doc_hashes],
        num_fewshot=parts[0].num_fewshot, limit=parts[0].limit,
        model=parts[0].model)
    rest = {name: row for name, row in samples.items() if name not in ARC}
    return {**rest, 'arc': merged}


def pick_run(payload, alloc):
    found = [run for run in payload['runs']
             if run['allocation']['name'] == alloc and ROUTER in run['routers']]
    if len(found) != 1:
        raise SystemExit(f'{alloc}+{ROUTER} の run が {len(found)} 個ある')
    return found[0]


def load_cell(alloc, task, seeds):
    """1マス（配分 × 校正タスク）の、seed ごとの {タスク: TaskSamples}。"""
    pattern = PATTERNS.get(alloc, DEFAULT_PATTERN)
    out = {}
    for seed in seeds:
        path = os.path.join(
            ROOT, (pattern % task).format(seed=seed, alloc=alloc))
        if not os.path.exists(os.path.join(path, 'summary.json')):
            raise SystemExit(f'{path} に summary.json が無い')
        with open(os.path.join(path, 'summary.json')) as handle:
            payload = json.load(handle)
        run = pick_run(payload, alloc)
        out[seed] = merge_arc(bench.load_samples(
            os.path.join(path, run['bench_samples'][ROUTER])))
    return out


def mean_metric(cell, task, metric, reference):
    values = [bench_stats.summarize(samples, reference)['tasks'][task][metric]
              for samples in cell.values()]
    return sum(values) / len(values)


def compare(base, cand, task, metric, reference):
    """1タスクの中で、seed を層としたペア付き再抽出。"""
    strata = [bench_stats.paired_differences(
        base[seed][task], cand[seed][task], metric,
        reference=reference.get(task) if reference else None)
        for seed in sorted(set(base) & set(cand))]
    return stratified_paired_bootstrap(
        strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
        unit='paired-question-within-seed',
        lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--metrics', default='acc,gold_nll')
    parser.add_argument('--allocs', default=','.join(ALLOCS))
    parser.add_argument('--reference', default=DENSE_REFERENCE)
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    metrics = [f.strip() for f in args.metrics.split(',') if f.strip()]
    allocs = [f.strip() for f in args.allocs.split(',') if f.strip()]
    reference = merge_arc(bench.load_samples(os.path.join(ROOT, args.reference)))
    dense = bench_stats.summarize(reference)

    cells = {(alloc, task): load_cell(alloc, task, seeds)
             for alloc in allocs for task in TASKS}

    print(f'seed {seeds} / N=8 A=6（スパース率25%）/ ルーター {ROUTER} / '
          f'校正 n=8 / 対角のみ（校正 = 評価タスクの train split）')
    print('問題数 ' + '  '.join(
        f'{SHORT[task]} {dense["n_docs"][task]}' for task in TASKS))

    for metric in metrics:
        higher = bench_stats.HIGHER_IS_BETTER[metric]
        print(f'\n== {metric}（行=配分、列=評価タスク。'
              f'{"高い" if higher else "低い"}ほど良い。* は列の最良）==')
        print(f'  {"":<8}{"Top-K":>6}' +
              ''.join(f'{SHORT[task]:>10}' for task in TASKS) +
              f'{"macro":>10}')
        values = {(alloc, task): mean_metric(cells[(alloc, task)], task,
                                             metric, reference)
                  for alloc in allocs for task in TASKS}
        best = {task: (max if higher else min)(
            allocs, key=lambda alloc: values[(alloc, task)])
            for task in TASKS}
        for alloc in allocs:
            row = ''.join(
                f'{values[(alloc, task)]:>9.4f}'
                f'{"*" if best[task] == alloc else " "}' for task in TASKS)
            macro = sum(values[(alloc, task)] for task in TASKS) / len(TASKS)
            topk = 6 - int(alloc.replace('uniform', ''))
            print(f'  {LABEL.get(alloc, alloc):<8}{topk:>6}{row}{macro:>10.4f}')
        dense_row = ''.join(
            f'{dense["tasks"][task][metric]:>9.4f} ' for task in TASKS)
        macro = sum(dense["tasks"][task][metric] for task in TASKS) / len(TASKS)
        print(f'  {"dense":<8}{"—":>6}{dense_row}{macro:>10.4f}')

    if BASELINE in allocs:
        print(f'\n== {LABEL[BASELINE]}（Top-K=0、枝刈りの退化ケース）との差'
              f'（列ごと。seed を層としたペア付き再抽出、{REPS} 回。'
              '* は95%区間が0をまたがない。正なら MoE 側が良い）==')
        for metric in metrics:
            print(f'\n  [{metric}]')
            sign = 1 if bench_stats.HIGHER_IS_BETTER[metric] else -1
            for task in TASKS:
                parts = []
                for alloc in allocs:
                    if alloc == BASELINE:
                        continue
                    got = compare(cells[(BASELINE, task)], cells[(alloc, task)],
                                  task, metric, reference)
                    mark = '*' if got['lower'] > 0 or got['upper'] < 0 else ' '
                    # 「大きいほど良い」向きに揃える。符号を返すと区間の
                    # 上下も入れ替わる
                    bounds = sorted((sign * got['lower'], sign * got['upper']))
                    parts.append(
                        f'{LABEL.get(alloc, alloc)} '
                        f'{sign * got["mean_difference"]:+.4f}'
                        f' [{bounds[0]:+.4f}, {bounds[1]:+.4f}]{mark}')
                print(f'    {SHORT[task]:<6} ' + '  '.join(parts))


if __name__ == '__main__':
    main()
