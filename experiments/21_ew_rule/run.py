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

CV のプローブは無ければこの中で測る（``--model`` ごとに別のプローブになる）。

  uv run python experiments/21_ew_rule/run.py --seed 0 --nactive 6
  uv run python experiments/21_ew_rule/run.py --seed 0 --nactive 4
  uv run python experiments/21_ew_rule/run.py --model mistral-7b --seed 0 --nactive 6
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
# 短い札 → HuggingFace の名前。experiments/25・28 と同じ辞書である
MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
}
# 出力先の名前に札が入らないモデル。report/21 の頃の名前を保つ
UNTAGGED_MODEL = 'llama2-7b'
# 取り込む dense。**モデルごとに測ったものを使う。** Llama は report/04 が
# 測ったもの（experiments/06・07 と同じ）、Mistral は experiments/25 の
# ``--check`` が測ったものである
DENSE_REFERENCES = {
    'llama2-7b': 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json',
    'mistral-7b': ('result_logs/exp25_check/bench_slimpajama_mistral-7b_seed0'
                   '/bench/dense.json'),
}
SMOKE_LAYERS = 2


def model_tag(model_key):
    """出力先に入るモデルの札。Llama-2-7b だけ空である。"""
    return '' if model_key == UNTAGGED_MODEL else f'_{model_key}'


def probe_dir(model_key, seed, smoke):
    suffix = '_smoke' if smoke else ''
    return (f'result_logs/ew_probe_{CALIB}{model_tag(model_key)}_signed'
            f'{suffix}_seed{seed}')


def ensure_probe(model_key, seed, probe_root, smoke):
    """CV のプローブが無ければここで測る。**符号付きのまま測る**（``--abs``
    を付けない）。絶対値を取ると CV がどの層でも τ を割り、規則が一様に縮退する
    （report/21）。

    **既にあるものは設定を照合してから使う。** smoke は2層・7本で測るので、
    そのまま本番が拾うと「32層のはずの配分が2層」になる。
    """
    out = Path(probe_root or probe_dir(model_key, seed, smoke))
    nsamples = 7 if smoke else NSAMPLES
    layers = SMOKE_LAYERS if smoke else None
    wanted = {'model': MODELS[model_key], 'calib': CALIB, 'seed': seed,
              'nsamples': nsamples, 'nexperts': NEXPERTS, 'layers': layers,
              'use_abs': False, 'sample_tokens': 128, 'seqlen': 2048}
    payload = ROOT / out / 'probe.json'
    if (ROOT / out / 'cv.npy').exists() and payload.exists():
        given = json.loads(payload.read_text()).get('arguments', {})
        gaps = [key for key in sorted(wanted) if given.get(key) != wanted[key]]
        if gaps:
            raise SystemExit(
                f'{out} は違う設定のプローブである（{", ".join(gaps)}）')
        log(f'  プローブ {out} : 済み')
        return str(out)
    command = ['uv', 'run', 'python',
               str(Path(__file__).resolve().parent / 'probe.py'),
               '--model', MODELS[model_key], '--calib', CALIB,
               '--seed', str(seed), '--nsamples', str(nsamples),
               '--nexperts', str(NEXPERTS), '--out', str(out)]
    if smoke:
        command += ['--layers', str(SMOKE_LAYERS)]
    log(f'$ {" ".join(command)}')
    subprocess.run(command, cwd=ROOT, check=True)
    return str(out)


def log(message=''):
    print(message, flush=True)


def ew_vector(probe_root, n_active):
    """プローブの CV から EW 既定値の配分を作る。

    ``probe.json`` の ``default`` をそのまま使わないのは、あれが A=6 で計算
    されているからである。A=4 ではクリップが変わりうるので、ここで引き直す。
    """
    array = np.load(ROOT / Path(probe_root) / 'cv.npy')
    cvs = [torch.from_numpy(row) for row in array]
    return ew_allocation(cvs, NEXPERTS, n_active, tau=DEFAULT_TAU,
                         alpha_min=DEFAULT_ALPHA_MIN,
                         alpha_max=DEFAULT_ALPHA_MAX)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--model', default='llama2-7b', choices=sorted(MODELS))
    parser.add_argument('--probe', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問。経路の確認')
    args = parser.parse_args(argv)

    probe = ensure_probe(args.model, args.seed, args.probe, args.smoke)
    values, detail = ew_vector(probe, args.nactive)
    if args.smoke:
        # --layers 2 で回すので、配分も先頭2層に切る（層数が合わないと弾かれる）
        values = values[:SMOKE_LAYERS]
    spec = ','.join(str(x) for x in values)
    # experiments/06・07 と同じ対照。先頭が cmoe run の対応比較の基準になる
    baseline = f'uniform{args.nactive // 2}'

    log(f'EW-rule  model={args.model} seed={args.seed} A={args.nactive} '
        f'τ={DEFAULT_TAU} α=[{DEFAULT_ALPHA_MIN}, {DEFAULT_ALPHA_MAX}]')
    log(f'  {spec}')
    log(f"  平均x={detail['mean_x']:.4f} "
        f"routing層={detail['n_routing_layers']}/{len(values)} "
        f"clip={detail['n_clipped']}")
    log(f'  対照（同じ run 内）= {baseline}')

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    smoke_tag = '_smoke' if args.smoke else ''
    out = (args.out or f'result_logs/ew_bench{model_tag(args.model)}{tag}'
           f'{smoke_tag}_seed{args.seed}')

    argv_cli = ['run', '--alloc', baseline, '--alloc', spec,
                '--model', MODELS[args.model],
                '--router', ROUTER, '--carver', CARVER,
                '--calib', CALIB, '--seeds', str(args.seed),
                '--nsamples', str(7 if args.smoke else NSAMPLES),
                '--nexperts', str(NEXPERTS), '--nactive', str(args.nactive),
                '--datasets', DATASETS, '--bench',
                '--bench-batch-size', str(8 if args.smoke else BENCH_BATCH)]
    if args.smoke:
        argv_cli += ['--bench-limit', '8', '--layers', str(SMOKE_LAYERS)]
    else:
        reference = ROOT / DENSE_REFERENCES[args.model]
        if not reference.exists():
            raise SystemExit(f'{reference} が無い。dense の基準が要る')
        argv_cli += ['--bench-reference', str(reference)]
    argv_cli += ['--out', out]

    log(f'\n$ uv run cmoe {" ".join(argv_cli)}\n')
    subprocess.run(['uv', 'run', 'cmoe', *argv_cli], cwd=ROOT, check=True)

    with (ROOT / out / 'ew_rule.json').open('w') as handle:
        json.dump({'seed': args.seed, 'model': args.model,
                   'model_name': MODELS[args.model], 'probe': probe,
                   'n_active': args.nactive,
                   'values': values, 'baseline': baseline, **detail},
                  handle, indent=1, ensure_ascii=False)
    log(f'\n配分の由来を {out}/ew_rule.json に書いた')
    return 0


if __name__ == '__main__':
    sys.exit(main())
