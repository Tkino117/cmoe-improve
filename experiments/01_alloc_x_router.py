"""層別配分 × ルーター方式の 2×2（report/13 の再現と拡張）。

report/13 は「2つの成果を同時に入れても効果は足し算にならない」を示した。
その表を1コマンドで出す。配分ごとに変換は1回で、ルーターは差し替えるだけ
なので、4構成でも変換は2回で済む。

  uv run python experiments/01_alloc_x_router.py
  uv run python experiments/01_alloc_x_router.py --carve-samples 64  # 未測定の条件

report/13 が積み残した切り分けが2つあり、どちらも引数を変えるだけで測れる。

* carve n=64 での再測定 … ビーム配分の C4 での優位が校正データ量に依存して
  いるかどうか（report/13 考察4）
* 方式6 との組み合わせ … C4 で唯一改善した方式（--routers に足す）
"""

import argparse
import sys

from cmoe.cli import main as cli_main

DEFAULTS = {
    'allocs': 'uniform3,beam',
    'routers': 'cmoe,oracle_recovery',
    'seeds': '0,1,2',
    'datasets': 'wikitext2,c4-new',
    'carve_samples': 8,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allocs', default=DEFAULTS['allocs'])
    parser.add_argument('--routers', default=DEFAULTS['routers'])
    parser.add_argument('--seeds', default=DEFAULTS['seeds'])
    parser.add_argument('--datasets', default=DEFAULTS['datasets'])
    parser.add_argument('--carve-samples', type=int, default=DEFAULTS['carve_samples'])
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    name = (f'alloc_x_router_n{args.carve_samples}_'
            f'{args.allocs.replace(",", "-")}_{args.routers.replace(",", "-")}')
    argv = [
        'run',
        '--alloc', args.allocs,
        '--router', args.routers,
        '--seeds', args.seeds,
        '--datasets', args.datasets,
        '--nsamples', str(args.carve_samples),
        '--out', args.out or f'result_logs/{name}',
    ]
    if args.diagnostics:
        argv.append('--diagnostics')
    # carve を増やすと捕捉が GPU に載らなくなるので、そのときだけバッチを分ける
    if args.carve_samples > 8:
        argv += ['--batch-chunk', '8']
    sys.exit(cli_main(argv))
