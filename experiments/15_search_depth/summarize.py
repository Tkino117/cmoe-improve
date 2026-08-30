"""15 の表を作る。深さ1 と深さ0 を 3 seed でまとめる。GPU は要らない。

段2 は seed ごとに、対照（幅2・深さ0）と候補（幅2・深さ1）を**同じ実行の中で**
測ってある。ここはその3実行を seed で束ね、CLI と同じ単位で再抽出する。

* PPL … seed を層とした、評価塊の対応再抽出
* ベンチ … (seed × タスク) を層とした、問題の対応再抽出

**どの run が対照でどれが候補かは、配分ベクトルで引く。** 探索が出した配分は
どちらも名前が `custom` なので、名前では区別できない。段1 の `search.json` が
持っている値と突き合わせることで、取り違えると必ず落ちるようにしてある。

  uv run python experiments/15_search_depth/summarize.py --seeds 0,1,2
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import compare_runs                                            # noqa: E402
from cmoe.eval import bench                                    # noqa: E402

BENCH = 'result_logs/bench_slimpajama_depth_seed{seed}'
DEPTH0 = 'result_logs/slimpajama_w2_n16_seed{seed}'
DEPTH1 = 'result_logs/slimpajama_w2L1_n16_seed{seed}'
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
ROUTER = 'cmoe'
DATASETS = ('wikitext2', 'c4-new')


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def searched(pattern, seed):
    """段1 の結果（配分・スコア・コスト）。"""
    record = load(os.path.join(pattern.format(seed=seed), 'search.json'))
    return {
        'values': list(record['allocation']['values']),
        'mean_x': record['allocation']['mean_x'],
        'score': record['score'],
        'spent': record['spent'],
        'calls': record['calls'],
        'seconds': record['seconds'],
        'lookahead': record['search'].get('lookahead') or 0,
    }


def runs_by_allocation(seed, wanted):
    """その seed のベンチ実行から、配分ベクトルで run を引く。"""
    directory = BENCH.format(seed=seed)
    payload = load(os.path.join(directory, 'summary.json'))
    found = [run for run in payload['runs']
             if list(run['allocation']['values']) == wanted]
    if len(found) != 1:
        raise SystemExit(
            f'{directory} に配分 {wanted} の run が {len(found)} 本ある')
    return os.path.join(ROOT, directory), found[0], payload


def collect(seeds):
    """(対照, 候補, 段1の記録)。compare_runs が食う形（seed -> (dir, run)）で返す。"""
    base, cand, searches = {}, {}, {}
    for seed in seeds:
        depth0, depth1 = searched(DEPTH0, seed), searched(DEPTH1, seed)
        if depth0['lookahead'] or depth1['lookahead'] != 1:
            raise SystemExit(f'seed {seed}: 深さが 0 と 1 の組になっていない')
        base[seed] = runs_by_allocation(seed, depth0['values'])[:2]
        cand[seed] = runs_by_allocation(seed, depth1['values'])[:2]
        searches[seed] = (depth0, depth1)
    return base, cand, searches


def show_searches(searches):
    print('== 探索（オラクル score。小さいほど良い）==')
    print(f'  {"seed":<6}{"深さ0":>12}{"深さ1":>12}{"差":>10}'
          f'{"平均x 0→1":>14}{"x が違う層":>12}{"コスト比":>10}')
    for seed, (depth0, depth1) in sorted(searches.items()):
        gap = (depth1['score'] - depth0['score']) / depth0['score'] * 100
        differing = sum(1 for left, right in zip(depth0['values'], depth1['values'])
                        if left != right)
        print(f'  {seed:<6}{depth0["score"]:>12.6f}{depth1["score"]:>12.6f}'
              f'{gap:>9.2f}%{depth0["mean_x"]:>7.2f}→{depth1["mean_x"]:<6.2f}'
              f'{differing:>8}/{len(depth0["values"])}'
              f'{depth1["spent"] / depth0["spent"]:>9.2f}倍')
    scores = [(d0['score'], d1['score']) for d0, d1 in searches.values()]
    gaps = [(b - a) / a * 100 for a, b in scores]
    print(f'  {"平均":<6}{sum(a for a, _ in scores) / len(scores):>12.6f}'
          f'{sum(b for _, b in scores) / len(scores):>12.6f}'
          f'{sum(gaps) / len(gaps):>9.2f}%')
    print('  深さ1 が全 seed で下回る' if all(gap < 0 for gap in gaps)
          else '  深さ1 が下回らない seed がある')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    args = parser.parse_args()
    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]

    base, cand, searches = collect(seeds)
    print(f'対照   幅2・深さ0（report/07 の探索）  候補   幅2・深さ1')
    print(f'seed   {seeds}\n')
    show_searches(searches)

    print(f'\n== PPL（{len(seeds)} seed 平均。差は共通評価塊の NLL、'
          '負なら候補が良い）==')
    for name, row in compare_runs.ppl_table(
            base, cand, ROUTER, ROUTER, DATASETS).items():
        print(f'  {name:<12} 対照 {row["base_mean_ppl"]:.6f} → '
              f'候補 {row["cand_mean_ppl"]:.6f}')
        compare_runs.show('NLL 差', row['paired'], 'mean_nll_difference')

    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    means, paired, n_seeds = compare_runs.bench_table(
        base, cand, ROUTER, ROUTER, reference)
    print(f'\n== ベンチ マクロ平均（{n_seeds} seed 平均）==')
    print(f'  {"":<16}{"対照":>12}{"候補":>12}')
    for metric, value in means['base']['macro'].items():
        print(f'  {metric:<16}{value:>12.6f}{means["cand"]["macro"][metric]:>12.6f}')
    print('\n== 対照比（* は95%区間が0をまたがない）==')
    for metric, row in paired.items():
        compare_runs.show(metric, row, 'mean_difference')

    print('\n== タスクごと acc（対照 → 候補）==')
    for task in means['base']['tasks']:
        print(f'  {task:<16}{means["base"]["tasks"][task]["acc"]:.4f}'
              f'→{means["cand"]["tasks"][task]["acc"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
