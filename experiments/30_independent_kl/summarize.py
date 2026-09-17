"""30 の集計。independent 対 幅1 を中心に、PPL と探索の score を並べる。

  uv run python experiments/30_independent_kl/summarize.py --seeds 0,1,2,3,4

* PPL は seed ごとの平均と、seed を層とした評価塊の対応再抽出（report/22 と同じ）
* 再現の検査: 幅1 の PPL を、以前の実行（同じ配分）と突き合わせる
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / 'result_logs'
REPS, BOOT_SEED = 10000, 20260813
DATASETS = ('wikitext2', 'c4-new')
LABELS = ('uniform', 'w1', 'w2', 'indep')
# 幅1 の PPL を以前に測った実行（A=6 だけ。A=4 の幅1 はここで初めて探した）
EARLIER_W1 = {0: 'ppl_slimpajama_w1_seed0', 1: 'bench_slimpajama_greedy_seed1',
              2: 'bench_slimpajama_greedy_seed2', 3: 'ppl_slimpajama_w1_seed3',
              4: 'ppl_slimpajama_w1_seed4'}


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def suffix(nactive):
    return '' if nactive == 6 else f'_a{nactive}'


def runs_by_vector(summary):
    return {tuple(run['allocation']['values']): run for run in summary['runs']}


def collect(seed, nactive):
    stages = load(LOGS / f'exp30_stages{suffix(nactive)}_seed{seed}.json')
    runs = runs_by_vector(load(ROOT / stages['ppl_dir'] / 'summary.json'))
    uniform = next(run for run in runs.values()
                   if run['allocation']['name'] == stages['uniform'])
    picked = {'uniform': uniform}
    for label, values in stages['vectors'].items():
        picked[label] = runs[tuple(values)]
    return stages, picked


def reproduce(seed, nactive, run):
    if nactive != 6 or seed not in EARLIER_W1:
        return None
    path = LOGS / EARLIER_W1[seed] / 'summary.json'
    if not path.exists():
        return None
    earlier = runs_by_vector(load(path)).get(tuple(run['allocation']['values']))
    if earlier is None:
        return None
    return max(abs(earlier['ppl']['cmoe'][name]['ppl'] - run['ppl']['cmoe'][name]['ppl'])
               for name in DATASETS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--out', default=str(LOGS / 'exp30_summary.json'))
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(',')]

    output = {}
    for nactive in (6, 4):
        found = []
        for seed in seeds:
            try:
                found.append((seed, *collect(seed, nactive)))
            except FileNotFoundError:
                pass
        if not found:
            continue
        sparsity = 100 * (8 - nactive) // 8
        print(f'\n## スパース率 {sparsity}%（A={nactive}）/ seed {[s for s, _, _ in found]}')

        table = {}
        print(f'{"配分":<8} {"平均 x":>7} {"WT2":>9} {"C4":>9} {"score":>10}')
        for label in LABELS:
            rows = [picked[label] for _, _, picked in found]
            mean_x = sum(row['allocation']['mean_x'] for row in rows) / len(rows)
            ppl = {name: sum(row['ppl']['cmoe'][name]['ppl'] for row in rows) / len(rows)
                   for name in DATASETS}
            scores = [stages['scores'].get(label) for _, stages, _ in found]
            score = (sum(scores) / len(scores)) if None not in scores else None
            table[label] = {'mean_x': mean_x, 'ppl': ppl, 'score': score}
            score_text = f'{score:>10.4f}' if score is not None else f'{"—":>10}'
            print(f'{label:<8} {mean_x:>7.3f} {ppl["wikitext2"]:>9.4f} '
                  f'{ppl["c4-new"]:>9.4f} {score_text}')

        paired = {}
        for base, cand in (('w1', 'indep'), ('uniform', 'indep'), ('uniform', 'w1'),
                           ('w2', 'indep')):
            paired[f'{cand}-{base}'] = {}
            for name in DATASETS:
                strata = [paired_differences(
                    SimpleNamespace(chunk_mean_nlls=picked[base]['ppl']['cmoe'][name]['chunk_mean_nlls']),
                    SimpleNamespace(chunk_mean_nlls=picked[cand]['ppl']['cmoe'][name]['chunk_mean_nlls']))
                    for _, _, picked in found]
                paired[f'{cand}-{base}'][name] = stratified_paired_bootstrap(
                    strata, reps=REPS, seed=BOOT_SEED)
        print('\nNLL の差（候補 − 基準、負なら候補が良い。* は 95% 区間が0をまたがない）')
        for key, per_dataset in paired.items():
            cells = []
            for name in DATASETS:
                result = per_dataset[name]
                low, high = result['lower'], result['upper']
                star = '*' if high < 0 or low > 0 else ''
                cells.append(f'{name} {result["mean_nll_difference"]:+.5f} '
                             f'[{low:+.5f}, {high:+.5f}]{star} '
                             f'{result["improved_seeds"]}/{result["n_seeds"]}')
            print(f'  {key:<14} ' + ' | '.join(cells))

        checks = {seed: reproduce(seed, nactive, picked['w1']) for seed, _, picked in found}
        checks = {seed: gap for seed, gap in checks.items() if gap is not None}
        if checks:
            print('\n再現の検査（幅1 の PPL、以前の実行との最大差）: '
                  + ', '.join(f'seed {seed} {gap:.3e}' for seed, gap in checks.items()))
        output[str(nactive)] = {
            'seeds': [seed for seed, _, _ in found], 'table': table,
            'paired': paired, 'reproduce_w1': checks,
            'vectors': {seed: stages['vectors'] for seed, stages, _ in found}}

    with Path(args.out).open('w') as handle:
        json.dump(output, handle, indent=1, ensure_ascii=False)
    print(f'\n{args.out} に書いた')


if __name__ == '__main__':
    main()
