"""11 の校正 × 評価の表を作る。GPU は要らない。

行は校正データ、列は評価タスク。1回の測定が5タスクすべてを採点しているので、
行を走らせれば表は埋まっている。ここが読むのは各 run が残した生の尤度で、
指標は ``bench_stats`` がそこから作る。

**ARC は既定で1タスクに束ねる。** ARC-e と ARC-c は同じ AI2-ARC を当時の
ベースライン2種が解けたかどうかで割ったもので、出題元・形式・語彙は同じ、
問題は重複しない。5×5 で見たとき両者は校正データとして交換可能だった
（ARC-e 列で ±0.0000、ARC-c 列で −0.0048、どちらも n.s.）ので、ExpertWeaver
Table 8 と同じく1タスクとして扱う。列は問題を突き合わせて束ねる（ARC-e 2,376
問 + ARC-c 1,172 問 = 3,548 問の1タスク）ので保存済みの尤度から作れるが、行の
方は校正トークンが変わるので測り直しである（`--calibs benchtrain:arc`）。
`--arc split` で分けたままの 5×5 に戻る。

対角（同じタスクで校正した行）が列で最良かを見る。列ごとの判定は、その列の
問題を対応づけ **seed を層とした**ペア付き再抽出で出す（compare_runs の
マクロ平均は (seed × タスク) を層にするので、列ごとの判定には使えない）。

  uv run python experiments/11_calib_task_matrix/summarize.py --seeds 0,1,2
  uv run python experiments/11_calib_task_matrix/summarize.py --arc split
"""

import argparse
import json
import os

from cmoe.eval import bench, bench_stats
from cmoe.eval.stats import stratified_paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPS, BOOT_SEED = 10000, 20260813
ALLOC, ROUTER = 'uniform4', 'cmoe'
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

# 束ねる順。この順で問題を並べるのは全行と dense で共通で、対応が崩れない
ARC = ('arc_easy', 'arc_challenge')
MERGED_TASKS = ('piqa', 'winogrande', 'arc', 'hellaswag')
SPLIT_TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')
SINGLE = 'result_logs/bench_calibtask_%s_seed{seed}'
# 5×5 の外側に置く2行。測り直していない
OUTSIDE = (('flanv2', 'result_logs/bench_calibtask_flanv2_seed{seed}'),
           ('mix(5)', 'result_logs/bench_benchtrain_seed{seed}'),
           ('wikitext2', 'result_logs/bench_h4/wikitext2_seed{seed}'))
SHORT = {'piqa': 'PIQA', 'winogrande': 'Wino', 'arc': 'ARC', 'arc_easy': 'ARC-e',
         'arc_challenge': 'ARC-c', 'hellaswag': 'HSwag'}
ROW_WIDTH = 13


def merge_arc(samples):
    """ARC-e と ARC-c を、問題を連結した1タスクにする。

    連結の順は ``ARC`` の並びで固定する。どの行も dense も同じ順で並べるので、
    問題ごとの対応（``doc_hashes``）はそのまま生きる。``harness_metrics`` は
    タスク単位の集計なので落とす。指標は生の尤度から作り直す。
    """
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


def pick_run(payload):
    """`uniform4` + `cmoe` の run。1ディレクトリに1つだけあるはず。"""
    found = [run for run in payload['runs']
             if run['allocation']['name'] == ALLOC and ROUTER in run['routers']]
    if len(found) != 1:
        raise SystemExit(f'{ALLOC}+{ROUTER} の run が {len(found)} 個ある')
    return found[0]


def load_row(pattern, seeds, merge):
    """1行ぶんの、seed ごとの {タスク: TaskSamples}。"""
    out = {}
    for seed in seeds:
        path = os.path.join(ROOT, pattern.format(seed=seed))
        if not os.path.exists(os.path.join(path, 'summary.json')):
            raise SystemExit(f'{path} に summary.json が無い')
        with open(os.path.join(path, 'summary.json')) as handle:
            payload = json.load(handle)
        run = pick_run(payload)
        samples = bench.load_samples(
            os.path.join(path, run['bench_samples'][ROUTER]))
        out[seed] = merge_arc(samples) if merge else samples
    return out


def means(row, reference):
    """seed 平均の {タスク: {指標: 値}}。"""
    per_seed = [bench_stats.summarize(samples, reference)
                for samples in row.values()]
    return {task: {metric: sum(one['tasks'][task][metric] for one in per_seed)
                   / len(per_seed)
                   for metric in per_seed[0]['tasks'][task]}
            for task in per_seed[0]['tasks']}


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


def table(label, table_rows, tasks, metric, higher_is_better):
    print(f'\n== {label}（行=校正、列=評価タスク。'
          f'{"高い" if higher_is_better else "低い"}ほど良い。'
          '● は対角、* は列の最良、◎ は両方）==')
    header = ''.join(f'{SHORT[task]:>9}' for task in tasks)
    print(f'  {"":<{ROW_WIDTH}}{header}{"macro":>9}')
    best = {}
    for task in tasks:
        values = [(row[metric][task], name) for name, row, _ in table_rows]
        best[task] = (max if higher_is_better else min)(values)[1]
    for name, row, diagonal in table_rows:
        cells = ''
        for task in tasks:
            on_diagonal, is_best = diagonal == task, best[task] == name
            mark = ('◎' if on_diagonal and is_best
                    else '●' if on_diagonal else '*' if is_best else ' ')
            cells += f'{row[metric][task]:>8.4f}{mark}'
        macro = sum(row[metric][task] for task in tasks) / len(tasks)
        print(f'  {name:<{ROW_WIDTH}}{cells}{macro:>9.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--metrics', default='acc,acc_norm,gold_nll,ref_kl')
    parser.add_argument('--arc', choices=('merge', 'split'), default='merge',
                        help='ARC-e と ARC-c を1タスクに束ねるか（既定は束ねる）')
    parser.add_argument('--reference', default=DENSE_REFERENCE)
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    metrics = [field.strip() for field in args.metrics.split(',') if field.strip()]
    merge = args.arc == 'merge'
    tasks = MERGED_TASKS if merge else SPLIT_TASKS
    reference = bench.load_samples(os.path.join(ROOT, args.reference))
    if merge:
        reference = merge_arc(reference)

    sources = ([(task, SINGLE % task, task) for task in tasks]
               + [(name, pattern, None) for name, pattern in OUTSIDE])
    loaded = [(name, load_row(pattern, seeds, merge), diagonal)
              for name, pattern, diagonal in sources]
    rows = [(name, {metric: {task: value[metric] for task, value in
                             means(row, reference).items()}
                    for metric in metrics}, diagonal)
            for name, row, diagonal in loaded]

    print(f'seed {seeds} / 配分 {ALLOC} 固定 / ルーター {ROUTER} / '
          f'N=8 A=6 / 校正 n=8 / ARC={args.arc}')
    dense = bench_stats.summarize(reference)
    print('dense  ' + '  '.join(
        f'{SHORT[task]} {dense["tasks"][task]["acc"]:.4f}' for task in tasks)
        + f'  macro {sum(dense["tasks"][t]["acc"] for t in tasks) / len(tasks):.4f}')
    print('問題数 ' + '  '.join(
        f'{SHORT[task]} {dense["n_docs"][task]}' for task in tasks))

    for metric in metrics:
        table(metric, rows, tasks, metric, bench_stats.HIGHER_IS_BETTER[metric])

    single = {diagonal: row for _, row, diagonal in loaded if diagonal}
    print('\n== 対角 − 他の行（列ごと。seed を層としたペア付き再抽出、'
          f'{REPS} 回。* は95%区間が0をまたがない）==')
    for metric in [m for m in metrics if m in ('acc', 'gold_nll')]:
        print(f'\n  [{metric}]')
        for task in tasks:
            cells = []
            for name, row, diagonal in loaded:
                if diagonal == task:
                    continue
                got = compare(row, single[task], task, metric, reference)
                mark = '*' if got['lower'] > 0 or got['upper'] < 0 else ' '
                cells.append(f'{SHORT.get(name, name)} '
                             f'{got["mean_difference"]:+.4f}'
                             f' [{got["lower"]:+.4f}, {got["upper"]:+.4f}]{mark}')
            print(f'    {SHORT[task]:<6} vs ' + '  '.join(cells))


if __name__ == '__main__':
    main()
