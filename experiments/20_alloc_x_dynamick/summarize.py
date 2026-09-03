"""20 の 2×2 を、seed をまたいでまとめる。GPU は要らない。

`cmoe run` が summary.json に書く対応のある比較は、どれも先頭構成
（`uniform3 + cmoe`）を基準にしたものである。この実験が答えたいのは**交互作用**
なので、必要なのは「同じ配分の中で可変Kに替えた差」と「同じルーターのまま配分を
替えた差」で、どちらも別の基準になる。ここはそれを組み直す。

再抽出の単位と層は CLI・`experiments/compare_runs.py` と同じ:

* PPL … seed を層とした、評価塊の対応再抽出
* ベンチ … (seed × タスク) を層とした、問題の対応再抽出

  uv run python experiments/20_alloc_x_dynamick/summarize.py --seeds 0,1,2
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))

from compare_runs import (bench_table, load, pick,  # noqa: E402
                          ppl_table, show)
from cmoe.eval import bench, bench_stats  # noqa: E402

# 配分の呼び名。`custom` は run が付ける名前で、中身は seed ごとの探索配分
ALLOCS = (('uniform3', 'uniform3'), ('uniform5', 'uniform5'),
          ('custom', '探索 幅4'))
ROUTERS = (('cmoe', '固定K'), ('dynamic_cmoe', '可変K'))
DENSE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
METRICS = ('acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement')


def compare(payloads, left, right, reference, datasets):
    """(配分, ルーター) の2点を比べる。left が対照。"""
    base = pick(payloads, left[0], left[1])
    cand = pick(payloads, right[0], right[1])
    ppl = ppl_table(base, cand, left[1], right[1], datasets)
    means, paired, n_seeds = bench_table(
        base, cand, left[1], right[1], reference)
    return ppl, means, paired, n_seeds


def live_budget(run):
    """K>0 の層だけで平均した予算 K。

    可変Kの実測（``mean_load``）は ``ThresholdRouter`` を持つ層だけの平均で、
    Top-K=0 の層は入らない。予算と並べるならこちらの分母に揃える必要がある
    （非一様な配分では `A - 平均x` とは別の数になる）。
    """
    allocation = run['allocation']
    total = allocation['n_active_total']
    live = [total - x for x in allocation['values'] if total - x > 0]
    return sum(live) / len(live)


def cell_name(alloc, router):
    alloc_label = dict(ALLOCS)[alloc]
    return f'{alloc_label} + {dict(ROUTERS)[router]}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--reference', default=str(ROOT / DENSE))
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    payloads = {}
    for seed in seeds:
        path = str(ROOT / 'result_logs' / f'bench_alloc_dynamic_seed{seed}')
        payloads[path] = load(path)
    datasets = payloads[list(payloads)[0]]['datasets']
    reference = bench.load_samples(args.reference)

    print(f'seed {seeds} / 校正 slimpajama n=16 / N=8 A=6')

    # --- 6セルの素の値 ------------------------------------------------------
    print(f'\n== 6セルの平均（{len(seeds)} seed）==')
    header = f'  {"構成":<24}{"wikitext2":>11}{"c4-new":>11}'
    header += ''.join(f'{m:>11}' for m in ('acc', 'gold_nll', 'ref_kl'))
    header += f'{"実測K":>9}'
    print(header)
    for alloc, _ in ALLOCS:
        for router, _ in ROUTERS:
            rows = pick(payloads, alloc, router)
            ppl = {name: sum(rows[s][1]['ppl'][router][name]['ppl']
                             for s in rows) / len(rows)
                   for name in datasets}
            samples = [bench.load_samples(
                os.path.join(rows[s][0], rows[s][1]['bench_samples'][router]))
                for s in sorted(rows)]
            macro = [bench_stats.summarize(one, reference)['macro']
                     for one in samples]
            mean = {m: sum(one[m] for one in macro) / len(macro)
                    for m in METRICS}
            loads = [rows[s][1]['realized_load'].get(f'{router}/bench')
                     for s in sorted(rows)]
            loads = [value for value in loads if value is not None]
            # 固定Kの行は実測を持たないので予算を出す。**分母を可変Kと
            # 揃える** — `mean_load` は K>0 の層だけで平均するので、
            # 全層で割った A - 平均x を並べると別の量になる
            mean_k = ((sum(loads) / len(loads)) if loads
                      else sum(live_budget(rows[s][1]) for s in rows)
                      / len(rows))
            print(f'  {cell_name(alloc, router):<24}'
                  f'{ppl["wikitext2"]:>11.4f}{ppl["c4-new"]:>11.4f}'
                  f'{mean["acc"]:>11.4f}{mean["gold_nll"]:>11.4f}'
                  f'{mean["ref_kl"]:>11.4f}{mean_k:>9.3f}')

    # --- 交互作用: 同じ配分の中で可変Kに替える ------------------------------
    print('\n== 可変Kの効き目（同じ配分の中。* は95%区間が0をまたがない）==')
    for alloc, label in ALLOCS:
        ppl, _, paired, n_seeds = compare(
            payloads, (alloc, 'cmoe'), (alloc, 'dynamic_cmoe'),
            reference, datasets)
        print(f'\n  [{label}]')
        for name in datasets:
            show(f'NLL {name}', ppl[name]['paired'], 'mean_nll_difference')
        for metric in METRICS:
            show(metric, paired[metric], 'mean_difference')

    # --- 交互作用: 同じルーターのまま配分を替える ---------------------------
    print('\n== 配分探索の効き目（uniform3 → 探索 幅4。同じルーターの中）==')
    for router, label in ROUTERS:
        ppl, _, paired, n_seeds = compare(
            payloads, ('uniform3', router), ('custom', router),
            reference, datasets)
        print(f'\n  [{label}]')
        for name in datasets:
            show(f'NLL {name}', ppl[name]['paired'], 'mean_nll_difference')
        for metric in METRICS:
            show(metric, paired[metric], 'mean_difference')

    # --- 全部を uniform3+固定K と比べる（どのセルが一番良いか）--------------
    print('\n== uniform3 + 固定K を基準にした5セル ==')
    for alloc, _ in ALLOCS:
        for router, _ in ROUTERS:
            if (alloc, router) == ('uniform3', 'cmoe'):
                continue
            ppl, _, paired, _ = compare(
                payloads, ('uniform3', 'cmoe'), (alloc, router),
                reference, datasets)
            print(f'\n  [{cell_name(alloc, router)}]')
            for name in datasets:
                show(f'NLL {name}', ppl[name]['paired'],
                     'mean_nll_difference')
            for metric in ('acc', 'gold_nll', 'ref_kl'):
                show(metric, paired[metric], 'mean_difference')

    # --- 実測の負荷 ---------------------------------------------------------
    print('\n== 走った routed expert 数（可変Kのみ。K>0 の層だけの平均）==')
    for alloc, label in ALLOCS:
        rows = pick(payloads, alloc, 'dynamic_cmoe')
        for seed in sorted(rows):
            run = rows[seed][1]
            values = run['realized_load']
            budget_values = run['allocation']['values']
            total = run['allocation']['n_active_total']
            topks = [total - x for x in budget_values]
            live = [k for k in topks if k > 0]
            budget = sum(live) / len(live)
            print(f'  {label:<10} seed {seed}  予算 {budget:.4f}  '
                  f'wt2 {values["dynamic_cmoe/wikitext2"]:.4f}  '
                  f'c4 {values["dynamic_cmoe/c4-new"]:.4f}  '
                  f'bench {values["dynamic_cmoe/bench"]:.4f}')


if __name__ == '__main__':
    main()
