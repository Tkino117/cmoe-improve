"""28 の表を作る。GPU は要らない。

列は WikiText-2 / C4 / 5タスクの正答率 / その平均。2段で出す。

段1（手法の構成どうし）… 原典の構造どうしを並べる
段2（分割だけの差）… 同じ配分どうしを並べる

対照（現行 CMoE の分割）は既存の `bench_slimpajama{,_a4}_seed<N>` から読む。

  uv run python experiments/28_llama_moe/table.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_runs                                          # noqa: E402
from run import DENSE_REFERENCE, NSAMPLES, POINTS, SEEDS, V2_SHARED  # noqa: E402

from cmoe.eval import bench, bench_stats                     # noqa: E402

TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')
HEADS = ('PIQA', 'WinoG.', 'ARC-e', 'ARC-c', 'HellaS.')
DATASETS = ('wikitext2', 'c4-new')
DENSE_PPL = 'result_logs/dense_ppl_llama-2-7b-hf/dense_ppl.json'
CONTROL = {6: 'result_logs/bench_slimpajama_seed{seed}',
           4: 'result_logs/bench_slimpajama_a4_seed{seed}'}


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def beam4(suffix, seed):
    record = load(f'result_logs/slimpajama{suffix}_w4_n{NSAMPLES}_seed{seed}/search.json')
    return ','.join(str(value) for value in record['allocation']['values'])


def pick_across_seeds(template, alloc_of_seed):
    """seed ごとにディレクトリも配分も違う run を1つに束ねる。"""
    rows = {}
    for seed in SEEDS:
        path = template.format(seed=seed)
        if not os.path.exists(os.path.join(ROOT, path, 'summary.json')):
            return None
        rows.update(compare_runs.pick(
            {path: load(os.path.join(path, 'summary.json'))},
            alloc_of_seed(seed), 'cmoe'))
    return rows


def cells(runs, reference):
    """PPL 2列 + タスク5列 + 平均。未測定なら空欄。"""
    if not runs:
        return ' | '.join(['—'] * (len(DATASETS) + len(TASKS) + 1))
    mean = lambda values: sum(values) / len(values)
    ppl = [mean([compare_runs.run_ppl(run, 'cmoe', name)['ppl']
                 for _, run in runs.values()]) for name in DATASETS]
    per_seed = [bench_stats.summarize(bench.load_samples(os.path.join(
        ROOT, path, compare_runs.run_bench_path(run, 'cmoe'))), reference)
        for path, run in runs.values()]
    tasks = [mean([one['tasks'][task]['acc'] for one in per_seed]) * 100
             for task in TASKS]
    average = mean([one['macro']['acc'] for one in per_seed]) * 100
    return ' | '.join([f'{ppl[0]:.2f}', f'{ppl[1]:.2f}']
                      + [f'{value:.2f}' for value in tasks]
                      + [f'{average:.2f}'])


def dense_cells(reference):
    dense = bench_stats.summarize(reference)
    payload = load(DENSE_PPL)['results']
    return ' | '.join(
        [f'{payload["wikitext2"]["ppl"]:.2f}', f'{payload["c4-new"]["ppl"]:.2f}']
        + [f'{dense["tasks"][task]["acc"] * 100:.2f}' for task in TASKS]
        + [f'{dense["macro"]["acc"] * 100:.2f}'])


def main():
    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    columns = ('| WikiText-2 | C4 | ' + ' | '.join(HEADS) + ' | Avg. |')
    rule = '--:|' * (len(DATASETS) + len(TASKS) + 1)

    for label, nactive, suffix in POINTS:
        control = CONTROL[nactive]
        v1 = f'result_logs/moe28_v1random_a{nactive}_seed{{seed}}'
        v2 = f'result_logs/moe28_v2_a{nactive}_seed{{seed}}'
        shared = f'uniform{V2_SHARED}'

        print()
        print(f'### スパース率 {label}（A={nactive}）')
        print()
        print('**手法の構成どうし**')
        print()
        print('| 手法 | 配分 ' + columns)
        print('|---|---|' + rule)
        for name, alloc_label, template, alloc_of_seed in (
                ('提案（beam 幅4）', 'beam 幅4', control,
                 lambda seed: beam4(suffix, seed)),
                ('LLaMA-MoE（Random）', 'uniform0', v1,
                 lambda seed: 'uniform0'),
                ('LLaMA-MoE-v2', shared, v2, lambda seed: shared)):
            runs = pick_across_seeds(template, alloc_of_seed)
            print(f'| {name} | {alloc_label} | {cells(runs, reference)} |')
        print(f'| dense（変換前） | — | {dense_cells(reference)} |')

        print()
        print('**分割だけの差（同じ配分どうし）**')
        print()
        print('| 配分 | 分割 ' + columns)
        print('|---|---|' + rule)
        for alloc_label, carver, template, alloc_of_seed in (
                ('uniform0', '現行 CMoE', control, lambda seed: 'uniform0'),
                ('uniform0', 'LLaMA-MoE（Random）', v1, lambda seed: 'uniform0'),
                (shared, '現行 CMoE', control, lambda seed: shared),
                (shared, 'LLaMA-MoE-v2', v2, lambda seed: shared),
                ('beam 幅4', '現行 CMoE', control,
                 lambda seed: beam4(suffix, seed)),
                ('beam 幅4', 'LLaMA-MoE（Random）', v1,
                 lambda seed: beam4(suffix, seed))):
            runs = pick_across_seeds(template, alloc_of_seed)
            print(f'| {alloc_label} | {carver} | {cells(runs, reference)} |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
