"""21-b プローブが保存した CV から配分ベクトルを出す。GPU は要らない。

``probe.py`` が書いた ``cv.npy`` は τ・α・A のどれにも依存しない。だから
しきい値の格子も、スパース率 25%/50% の違いも、ここで CPU だけで出せる。
GPU を使うのは層ごとの CV を作る1回だけである。

  # EW 既定値で、両スパース率の配分を出す
  uv run python experiments/21_ew_rule/allocate.py --seed 0

  # τ の格子。縮退しない範囲を探す
  uv run python experiments/21_ew_rule/allocate.py --seed 0 --grid

``--grid`` が出すのは候補の一覧であって、選んだ1本ではない。**どれを使うかは
校正データ上の目的関数（``cmoe score --oracle suffix_kl``）で決める** —
評価ベンチを見て選ぶと、EW の既定値が評価ベンチ上の掃引で決まっていることを
批判している側が同じことをすることになる。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from cmoe.alloc.ew_rule import (DEFAULT_ALPHA_MAX, DEFAULT_ALPHA_MIN,
                                DEFAULT_TAU, ew_allocation)

# τ の格子。EW 既定の 0.6 を含み、下は CV 分布の中央値あたりまで下ろす
TAU_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80)
# (α_min, α_max)。既定・全域・狭の3通り
ALPHA_GRID = ((0.2, 0.7), (0.0, 1.0), (0.3, 0.6))


def load_cv(path):
    array = np.load(path)
    return [torch.from_numpy(row) for row in array]


def describe(values, detail, n_active):
    return (f"平均x={detail['mean_x']:.4f} "
            f"routing層={detail['n_routing_layers']}/{len(values)} "
            f"clip={detail['n_clipped']} "
            f"異なるx={sorted(set(values))}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ew-allocate')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--probe', default=None, help='プローブの出力先')
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=None,
                        help='省略時は 6 と 4 の両方を出す')
    parser.add_argument('--grid', action='store_true', help='τ・α を振る')
    parser.add_argument('--out', default=None, help='JSON の書き出し先')
    args = parser.parse_args(argv)

    probe = args.probe or f'result_logs/ew_probe_{args.calib}_seed{args.seed}'
    cvs = load_cv(os.path.join(probe, 'cv.npy'))
    actives = [args.nactive] if args.nactive else [6, 4]

    stacked = torch.stack(cvs)
    print(f'CV: {tuple(stacked.shape)} (層 × ニューロン)  '
          f'全体中央値={stacked.median():.3f} 95%={stacked.quantile(0.95):.3f} '
          f'max={stacked.max():.3f}')

    rows = []
    combinations = ([(tau, lo, hi) for tau in TAU_GRID for lo, hi in ALPHA_GRID]
                    if args.grid else
                    [(DEFAULT_TAU, DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX)])
    for n_active in actives:
        print(f'\n=== A={n_active}（スパース率 {100 - 100 * n_active // 8}%）===')
        for tau, lo, hi in combinations:
            values, detail = ew_allocation(cvs, args.nexperts, n_active,
                                           tau=tau, alpha_min=lo, alpha_max=hi)
            flat = len(set(values)) == 1
            mark = ' ** 縮退（一様配分）' if flat else ''
            print(f'  τ={tau:<5} α=[{lo}, {hi}]  {describe(values, detail, n_active)}{mark}')
            print(f'    {",".join(str(x) for x in values)}')
            rows.append({'n_active': n_active, 'tau': tau, 'alpha_min': lo,
                         'alpha_max': hi, 'values': values,
                         'degenerate': flat, **detail})

    if args.out:
        with open(args.out, 'w') as handle:
            json.dump({'probe': probe, 'rows': rows}, handle, indent=1,
                      ensure_ascii=False)
        print(f'\n書いた: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
