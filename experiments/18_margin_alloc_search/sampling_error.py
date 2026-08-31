"""目的関数を n 問で推定したときの標準誤差。GPU は要らない。

段3 が残した問題ごと・選択肢ごとの生の対数尤度（``bench/*.json``）を読み戻し、
校正と同じ式で問題ごとの M を作る:

    M(q) = logsumexp( ll(不正解) ) - ll(正解)      （β = 1）

配分 A と B の**同じ問題での差** d(q) = M_A(q) - M_B(q) を、タスクごとに n 問ずつ
復元抽出してマクロ平均する操作を繰り返し、その標準偏差を取る。

**測っているのは1つだけである** — 「モデルを固定したまま、採点に使う問題だけを
引き直したときに、測定値がどれだけ動くか」。校正を引き直せば expert の切り出しも
変わるが、それはここに入っていない。また読んでいるのはベンチの評価 split で、
校正が引いているのは train split である。

生成器は**呼び出しごとに同じ seed で作り直す**。使い回すと同じ量を2度計算した
ときに値がずれ、どちらを引くべきか分からなくなる。

    uv run python experiments/18_margin_alloc_search/sampling_error.py
"""

import argparse
import json
import math
from pathlib import Path

import torch

from cmoe.eval import bench

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')
# 段3 が測った順。``summary.json`` の configurations と同じ並び
NAMES = ('uniform4', 'uniform3', 'w2', 'w3', 'w4')
REPS = 100_000
SEED = 20260813


def per_document_margin(samples):
    """問題ごとの M（β=1）。校正のオラクルと同じ式。"""
    values = []
    for lls, gold in zip(samples.loglikelihoods, samples.gold):
        wrong = [ll for index, ll in enumerate(lls) if index != gold]
        top = max(wrong)
        values.append(top + math.log(sum(math.exp(v - top) for v in wrong))
                      - lls[gold])
    return torch.tensor(values, dtype=torch.float64)


def load(out_dir):
    """段3 の出力から、配分ごと・タスクごとの M を読む。"""
    runs = json.load(open(out_dir / 'summary.json'))['bench_summary']['configurations']
    if len(runs) != len(NAMES):
        raise SystemExit(f'{out_dir} の構成が {len(runs)} 個。{len(NAMES)} 個のはず')
    loaded, allocations = {}, {}
    for index, (name, run) in enumerate(zip(NAMES, runs)):
        samples = bench.load_samples(out_dir / 'bench' / f'run{index:03d}_cmoe.json')
        loaded[name] = {task: per_document_margin(samples[task]) for task in TASKS}
        spec = run['allocation']
        allocations[name] = ([int(v) for v in spec.split(',')] if ',' in spec
                             else [int(spec[-1])] * 32)
    return loaded, allocations, {task: runs and samples[task].n_docs for task in TASKS}


def difference(margins, left, right, per_task, reps=REPS, seed=SEED):
    """(全件での差, n 問で推定したときの差の標準偏差)。差の向きは left − right。"""
    generator = torch.Generator().manual_seed(seed)
    boot = torch.zeros(reps, dtype=torch.float64)
    whole = 0.0
    for task in TASKS:
        values = margins[left][task] - margins[right][task]
        whole += float(values.mean()) / len(TASKS)
        index = torch.randint(values.numel(), (reps, per_task), generator=generator)
        boot += values[index].mean(dim=1) / len(TASKS)
    return whole, float(boot.std())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='result_logs/bench_margin_seed0',
                        help='段3 の出力ディレクトリ')
    parser.add_argument('--reps', type=int, default=REPS)
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out
    margins, allocations, n_docs = load(out_dir)
    print(f'reps={args.reps:,} / seed={SEED} / 問題数 ' +
          ' '.join(f'{t}={n_docs[t]}' for t in TASKS))

    print('\n=== 評価 split 全件での macro M（小さいほど良い）===')
    for name in NAMES:
        whole = sum(float(margins[name][t].mean()) for t in TASKS) / len(TASKS)
        print(f'  {name:10} {whole:+.4f}')

    print('\n=== uniform4 との差と、125問で推定したときの差の SD ===')
    for name in NAMES[1:]:
        whole, sd = difference(margins, name, 'uniform4', 25, args.reps)
        print(f'  {name:10} 差={whole:+.4f}  SD={sd:.4f}')

    print('\n=== 全10組（差の向きは、表の左 − 右）===')
    print(f"  {'組':24} {'違う層数':>8} {'|Δx|合計':>9} {'差':>9} {'SD':>8}")
    for i, left in enumerate(NAMES):
        for right in NAMES[i + 1:]:
            delta = [a - b for a, b in zip(allocations[left], allocations[right])]
            whole, sd = difference(margins, left, right, 25, args.reps)
            print(f'  {left + " 対 " + right:24} {sum(1 for v in delta if v):8d} '
                  f'{sum(abs(v) for v in delta):9d} {whole:+9.4f} {sd:8.4f}')

    print('\n=== 問題数を変えたときの SD（w4 対 uniform4）===')
    for per_task in (25, 100, 400, 1600, 6400):
        _, sd = difference(margins, 'w4', 'uniform4', per_task, args.reps)
        print(f'  {per_task:5d}問/タスク（計{per_task * len(TASKS):6d}問）  SD={sd:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
