"""21-c EW-rule の配分を、report/09・16 と同じ土俵でベンチにかける。

``probe.py`` が保存した CV から EW の式(4)(5)(6) で配分を作り、``cmoe run`` で
PPL と選択問題を測る。**校正・動作点・分割・ルーター・評価・dense の基準は
experiments/06（25%）・07（50%）と同一**で、動かしたのは配分だけである。

対照の一様配分を毎回**同じ run に同梱する**のには2つ理由がある。

* run 内の対応再抽出がそのまま出る（先頭の配分が ``cmoe run`` の基準になる）
* **既存の測定を再現しているかの検査になる。** 作業ツリーには report/19・20 で
  触った ``assemble.py`` / ``moe/modules.py`` / ``router/registry.py`` の変更が
  ある。これが現行 CMoE の挙動を動かしていたら、``bench_slimpajama_*`` との
  run をまたぐ比較は成り立たない。同じ一様配分の数字が一致することを見てから
  でないと、EW-rule の差を読んではいけない

  uv run python experiments/21_ew_rule/run.py --seed 0 --nactive 6
  uv run python experiments/21_ew_rule/run.py --seed 0 --nactive 4
  uv run python experiments/21_ew_rule/run.py --seed 0 --nactive 6 --smoke
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cmoe.alloc.ew_rule import (DEFAULT_ALPHA_MAX, DEFAULT_ALPHA_MIN,
                                DEFAULT_TAU, ew_allocation)

ROOT = Path(__file__).resolve().parents[2]

CALIB = 'slimpajama'
NSAMPLES = 16
ROUTER = 'cmoe'
CARVER = 'cmoe'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
NEXPERTS = 8
# report/04 が測った dense。experiments/06・07 と同じものを取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
SMOKE_LAYERS = 2


def log(message=''):
    print(message, flush=True)


def ew_vector(seed, n_active, probe_root=None):
    """プローブの CV から EW 既定値の配分を作る。

    ``probe.json`` の ``default`` をそのまま使わないのは、あれが A=6 で計算
    されているからである。A=4 ではクリップが変わりうるので、ここで引き直す。
    """
    probe = Path(probe_root or
                 f'result_logs/ew_probe_{CALIB}_signed_seed{seed}')
    array = np.load(ROOT / probe / 'cv.npy')
    cvs = [torch.from_numpy(row) for row in array]
    return ew_allocation(cvs, NEXPERTS, n_active, tau=DEFAULT_TAU,
                         alpha_min=DEFAULT_ALPHA_MIN,
                         alpha_max=DEFAULT_ALPHA_MAX)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--probe', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問。経路の確認')
    args = parser.parse_args(argv)

    values, detail = ew_vector(args.seed, args.nactive, args.probe)
    if args.smoke:
        # --layers 2 で回すので、配分も先頭2層に切る（層数が合わないと弾かれる）
        values = values[:SMOKE_LAYERS]
    spec = ','.join(str(x) for x in values)
    # experiments/06・07 と同じ対照。先頭が cmoe run の対応比較の基準になる
    baseline = f'uniform{args.nactive // 2}'

    log(f'EW-rule  seed={args.seed} A={args.nactive} '
        f'τ={DEFAULT_TAU} α=[{DEFAULT_ALPHA_MIN}, {DEFAULT_ALPHA_MAX}]')
    log(f'  {spec}')
    log(f"  平均x={detail['mean_x']:.4f} "
        f"routing層={detail['n_routing_layers']}/{len(values)} "
        f"clip={detail['n_clipped']}")
    log(f'  対照（同じ run 内）= {baseline}')

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    out = args.out or f'result_logs/ew_bench{tag}_seed{args.seed}'

    argv_cli = ['run', '--alloc', baseline, '--alloc', spec,
                '--router', ROUTER, '--carver', CARVER,
                '--calib', CALIB, '--seeds', str(args.seed),
                '--nsamples', str(7 if args.smoke else NSAMPLES),
                '--nexperts', str(NEXPERTS), '--nactive', str(args.nactive),
                '--datasets', DATASETS, '--bench',
                '--bench-batch-size', str(8 if args.smoke else BENCH_BATCH)]
    if args.smoke:
        argv_cli += ['--bench-limit', '8', '--layers', str(SMOKE_LAYERS)]
    else:
        reference = ROOT / DENSE_REFERENCE
        if not reference.exists():
            raise SystemExit(f'{reference} が無い。dense の基準が要る')
        argv_cli += ['--bench-reference', str(reference)]
    argv_cli += ['--out', out]

    log(f'\n$ uv run cmoe {" ".join(argv_cli)}\n')
    subprocess.run(['uv', 'run', 'cmoe', *argv_cli], cwd=ROOT, check=True)

    with (ROOT / out / 'ew_rule.json').open('w') as handle:
        json.dump({'seed': args.seed, 'n_active': args.nactive,
                   'values': values, 'baseline': baseline, **detail},
                  handle, indent=1, ensure_ascii=False)
    log(f'\n配分の由来を {out}/ew_rule.json に書いた')
    return 0


if __name__ == '__main__':
    sys.exit(main())
