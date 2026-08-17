"""層ごとの x を探す。オラクルと探索を別々に選ぶ。

report/06 の成果1（ビーム配分）を出したのと同じ目的関数・同じ幅が既定である。
違うのは、オラクルと探索が別々に差し替えられることで、「計算量を落とす探索」の
比較がここから始まる。

  uv run python experiments/02_alloc_search.py                    # 既定（数時間）
  uv run python experiments/02_alloc_search.py --layers 2         # 配線確認
  uv run python experiments/02_alloc_search.py --oracle local_error
  uv run python experiments/02_alloc_search.py --search greedy    # 幅1、貪欲

オラクルは安い順に3段ある。

* ``mass`` / ``mass_squared`` … 層ローカル。取りこぼした活性質量の割合
* ``local_error``             … 層ローカル。その層の相対二乗出力誤差 L(x)
* ``suffix_kl``               … 残りを dense のまま走らせた出力分布の KL（既定）

層ローカルの2つは後続の層を1つも走らせないので桁違いに安い。それで
``suffix_kl`` と同じ配分に辿り着けるかが、そのまま実験になる — 安いオラクルが
選択を保つなら、探索のコストはそこで落ちる。

出てくる配分は ``run --alloc`` にそのまま貼れる。探索と評価を分けてあるのは、
探索が数時間かかる一方で、出た配分を使う実験はその後何度も走るからである。
"""

import argparse
import sys

from cmoe.cli import main as cli_main

DEFAULTS = {
    # report/06 のビーム配分と同じ目的関数・同じ幅
    'oracle': 'suffix_kl',
    'search': 'beam',
    'width': 4,
    'nsamples': 8,
    'seed': 0,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', default=DEFAULTS['oracle'])
    parser.add_argument('--search', default=DEFAULTS['search'])
    # None なら探索ごとの既定（beam は4、greedy は1）。--search greedy に
    # 幅4 が付いて回ると、それだけで拒まれる
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--nsamples', type=int, default=DEFAULTS['nsamples'])
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    parser.add_argument('--budget', type=float, default=None)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    width = DEFAULTS['width'] if args.width is None and args.search == 'beam' \
        else args.width
    name = (f'search_{args.search}{width or ""}_{args.oracle}_'
            f'n{args.nsamples}_seed{args.seed}')
    argv = [
        'search',
        '--oracle', args.oracle,
        '--search', args.search,
        '--nsamples', str(args.nsamples),
        '--seed', str(args.seed),
        '--out', args.out or f'result_logs/{name}',
    ]
    if width is not None:
        argv += ['--width', str(width)]
    if args.budget is not None:
        argv += ['--budget', str(args.budget)]
    if args.layers is not None:
        argv += ['--layers', str(args.layers)]
    # carve を増やすと捕捉が GPU に載らなくなるので、そのときだけバッチを分ける
    if args.nsamples > 8:
        argv += ['--batch-chunk', '8']
    sys.exit(cli_main(argv))
