"""26 の表を作る。幅1 を、同じ校正の幅2/3/4 と並べる。GPU は要らない。

主判定は探索の score だけで付く（幅1〜4 は同じ校正トークンの上で測ってあるので
直接比べられる）。ベンチと PPL は記述用で、段2 の3構成を seed で束ねる。

**run をまたぐ前に再現の検査を通す。** 段2 に同梱した一様（A=6 なら uniform3）と、
既存の `bench_slimpajama_seed<N>` の同じ配分を突き合わせ、Δ=0 でなければ止まる。
report/22 と同じ理由で、これが合わないうちは run をまたぐ比較を読んではいけない。

  uv run python experiments/26_greedy_slimpajama/summarize.py --seeds 0,1,2
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_runs                                            # noqa: E402
from run import (CALIB, CONTROL_WIDTH, DENSE_REFERENCE, NSAMPLES,  # noqa: E402
                 ROUTER, WIDTH, suffix)

from cmoe.eval import bench, bench_stats                       # noqa: E402

WIDTHS = (1, 2, 3, 4)
DATASETS = ('wikitext2', 'c4-new')
# 再現の検査の相手。report/06・07（A=6）と report/09（A=4）の測定
EXISTING_BENCH = {6: 'result_logs/bench_slimpajama_seed{seed}',
                  4: 'result_logs/bench_slimpajama_a4_seed{seed}'}
TOLERANCE = 0.0


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def search_record(seed, width, nactive):
    path = (f'result_logs/{CALIB}{suffix(nactive)}_w{width}'
            f'_n{NSAMPLES}_seed{seed}/search.json')
    if not os.path.exists(os.path.join(ROOT, path)):
        raise SystemExit(f'{path} が無い')
    record = load(path)
    return {'values': list(record['allocation']['values']),
            'mean_x': record['allocation']['mean_x'],
            'score': record['score'], 'spent': record['spent'],
            'calls': record['calls'], 'seconds': record['seconds'],
            'recheck_gap': record['recheck']['gap']}


def run_by_allocation(directory, wanted):
    """ベンチ実行から run を1本引く。名前ではなく配分ベクトルで引く。

    探索が出した配分はどれも名前が `custom` なので、名前では区別できない。
    """
    payload = load(os.path.join(directory, 'summary.json'))
    if isinstance(wanted, str):
        found = [run for run in payload['runs']
                 if run['allocation']['name'] == wanted]
    else:
        found = [run for run in payload['runs']
                 if list(run['allocation']['values']) == list(wanted)]
    if len(found) != 1:
        raise SystemExit(f'{directory} に配分 {wanted} の run が {len(found)} 本ある')
    return os.path.join(ROOT, directory), found[0]


def show_searches(seeds, searches):
    """主判定。同じ校正トークンの上なので、seed の行の中では横に読める。"""
    print('== 探索（オラクル score、小さいほど良い。行の中でだけ比べられる）==')
    header = ''.join(f'{"幅" + str(w):>14}' for w in WIDTHS)
    print(f'  {"seed":<6}{header}{"幅1 の順位":>12}')
    ranks = []
    for seed in seeds:
        row = searches[seed]
        scores = [row[w]['score'] for w in WIDTHS]
        best = min(scores)
        cells = ''.join(
            f'{("**" if s == best else "  ") + f"{s:.6f}":>14}' for s in scores)
        rank = 1 + sum(1 for s in scores if s < row[WIDTH]['score'])
        ranks.append(rank)
        print(f'  {seed:<6}{cells}{f"{rank}/{len(WIDTHS)}":>12}')
    print(f'  {"平均x":<6}' + ''.join(
        f'{sum(searches[s][w]["mean_x"] for s in seeds) / len(seeds):>14.3f}'
        for w in WIDTHS))
    print(f'  {"コスト":<6}' + ''.join(
        f'{searches[seeds[0]][w]["spent"]:>14g}' for w in WIDTHS))
    print(f'  {"分":<6}' + ''.join(
        f'{sum(searches[s][w]["seconds"] for s in seeds) / len(seeds) / 60:>14.1f}'
        for w in WIDTHS))
    worst = all(rank == len(WIDTHS) for rank in ranks)
    print(f'\n  幅{WIDTH} は {len(seeds)} seed'
          + ('すべてで最下位（H1）' if worst else 'のうち最下位でない seed がある（H0）'))
    gaps = [(searches[s][WIDTH]['score'] - searches[s][CONTROL_WIDTH]['score'])
            / searches[s][CONTROL_WIDTH]['score'] * 100 for s in seeds]
    print(f'  幅{WIDTH} は幅{CONTROL_WIDTH} より '
          + ' / '.join(f'{gap:+.2f}%' for gap in gaps)
          + f'（平均 {sum(gaps) / len(gaps):+.2f}%）')
    print('  測り直しの差 '
          + ' '.join(f'{searches[s][WIDTH]["recheck_gap"]:.1e}' for s in seeds))


def recheck(seeds, nactive, uniform):
    """再現の検査。段2 の一様と、既存の測定の同じ一様が一致するか。"""
    print(f'\n== 再現の検査（{uniform}。既存の測定との差。0 でなければ止まる）==')
    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    for seed in seeds:
        new_dir, new_run = run_by_allocation(
            f'result_logs/bench_{CALIB}_greedy{suffix(nactive)}_seed{seed}',
            uniform)
        old_dir, old_run = run_by_allocation(
            EXISTING_BENCH[nactive].format(seed=seed), uniform)
        gaps = []
        for name in DATASETS:
            gaps.append((name, new_run['ppl'][ROUTER][name]['mean_nll']
                         - old_run['ppl'][ROUTER][name]['mean_nll']))
        new_acc, old_acc = (
            bench_stats.summarize(bench.load_samples(os.path.join(
                path, run['bench_samples'][ROUTER])), reference)['macro']['acc']
            for path, run in ((new_dir, new_run), (old_dir, old_run)))
        gaps.append(('bench acc', new_acc - old_acc))
        print(f'  seed {seed}: '
              + '  '.join(f'{name} Δ={gap:+.3e}' for name, gap in gaps))
        if any(abs(gap) > TOLERANCE for _, gap in gaps):
            raise SystemExit(
                f'seed {seed} の {uniform} が既存の測定と一致しない。'
                'run をまたぐ比較を読んではいけない')
    print('  すべて Δ=0。run をまたいで読んでよい')


def compare(label, base, cand, reference):
    print(f'\n== {label} ==')
    for name, row in compare_runs.ppl_table(
            base, cand, ROUTER, ROUTER, DATASETS).items():
        print(f'  {name:<12} 対照 {row["base_mean_ppl"]:.6f} → '
              f'候補 {row["cand_mean_ppl"]:.6f}')
        compare_runs.show('NLL 差', row['paired'], 'mean_nll_difference')
    means, paired, n_seeds = compare_runs.bench_table(
        base, cand, ROUTER, ROUTER, reference)
    print(f'  ベンチ マクロ平均（{n_seeds} seed）')
    print(f'  {"":<16}{"対照":>12}{"候補":>12}')
    for metric, value in means['base']['macro'].items():
        print(f'  {metric:<16}{value:>12.6f}'
              f'{means["cand"]["macro"][metric]:>12.6f}')
    print('  対照比（* は95%区間が0をまたがない）')
    for metric, row in paired.items():
        compare_runs.show(metric, row, 'mean_difference')
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--no-bench', action='store_true',
                        help='探索の表だけ出す（段2 を回していないとき）')
    args = parser.parse_args()
    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]

    searches = {seed: {width: search_record(seed, width, args.nactive)
                       for width in WIDTHS} for seed in seeds}
    print(f'校正 {CALIB} n={NSAMPLES} / N=8 A={args.nactive} / seed {seeds}\n')
    show_searches(seeds, searches)
    if args.no_bench:
        return 0

    uniform = f'uniform{args.nactive // 2}'
    recheck(seeds, args.nactive, uniform)

    bench_dir = f'result_logs/bench_{CALIB}_greedy{suffix(args.nactive)}_seed{{seed}}'
    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    picks = {}
    for key, wanted in (('greedy', WIDTH), ('control', CONTROL_WIDTH),
                        ('uniform', None)):
        picks[key] = {
            seed: run_by_allocation(
                bench_dir.format(seed=seed),
                uniform if wanted is None else searches[seed][wanted]['values'])
            for seed in seeds}

    compare(f'対照 幅{CONTROL_WIDTH} → 候補 幅{WIDTH}（同じ実行の中で測った）',
            picks['control'], picks['greedy'], reference)
    means = compare(f'対照 {uniform} → 候補 幅{WIDTH}',
                    picks['uniform'], picks['greedy'], reference)
    print('\n== タスクごと acc（幅1）==')
    for task, row in means['cand']['tasks'].items():
        print(f'  {task:<16}{row["acc"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
