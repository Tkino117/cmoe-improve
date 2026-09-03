"""21-a ExpertWeaver の配分規則が、この動作点で何を出すかを先に見る。

**ベンチを回す前に必ずここを通す。** 規則は
``r_ℓ = |{CV_i > τ}| / d_ffn`` → ``α_ℓ`` → ``x_ℓ`` という写像で、τ が CV の分布
から外れていると **r_ℓ が全層 0 か 1 に張り付く**。そうなると x_ℓ は全層同じ値
になり、EW-rule はただの一様配分に潰れる。潰れた配分でベンチを数時間回しても、
分かるのは「一様配分をもう一度測った」ことだけである。

ここが出すのは層ごとの CV（``cv.npy``）と、EW 既定値での r_ℓ / x_ℓ である。
**CV は τ・α・A のどれにも依存しない**ので、この1回の前向きから、しきい値の
格子もスパース率の違いも CPU だけで出せる（``allocate.py``）。

  uv run python experiments/21_ew_rule/probe.py --seed 0
  uv run python experiments/21_ew_rule/probe.py --seed 0 --layers 2   # 動作確認

校正・動作点は report/09・16（= res09 / res16）と揃えてある。slimpajama
n=16 × 2048、N=8。CV はスパース率に依存しないので、A はここでは表示にしか
使わない。
"""

import argparse
import os
import sys

import numpy as np
import torch

from cmoe import runlog
from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.alloc.ew_rule import (DEFAULT_ALPHA_MAX, DEFAULT_ALPHA_MIN,
                                DEFAULT_TAU, ew_allocation, layer_gate_profile,
                                neuron_cv)
from cmoe.data.registry import load_calibration

log = runlog.log


@torch.no_grad()
def dense_ffn_forward(dense, z, residual, batch_chunk=None, device=None):
    """dense の FFN を通して次の層の入力を作る。

    EW は **dense モデルの**活性パターンを読む（§3.1）。変換済みの接頭辞の上で
    測るのは別の量なので、ここは最後まで dense で進める。
    """
    step = batch_chunk or z.shape[0]
    rows = []
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step]
        base = residual[start:start + step]
        if device is not None:
            chunk, base = chunk.to(device), base.to(device)
        out = dense.down_proj(dense.act_fn(dense.gate_proj(chunk))
                              * dense.up_proj(chunk))
        rows.append((base + out).to(residual.device))
        del out, chunk, base
    return torch.cat(rows, dim=0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ew-probe')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=16,
                        help='校正の系列数。トークン総量は他の行と揃える')
    parser.add_argument('--sample-tokens', type=int, default=128,
                        help='1サンプルの長さ。M = nsamples × seqlen / これ。'
                             '既定 128 は M=256 で EW の 240 に対応する')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=6,
                        help='表示用。CV はこれに依存しない')
    parser.add_argument('--batch-chunk', type=int, default=None)
    parser.add_argument('--layers', type=int, default=None,
                        help='先頭 n 層だけ。動作確認用')
    parser.add_argument('--abs', dest='use_abs', action='store_true',
                        help='活性の絶対値で測る。EW 式から外れる側')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    out = args.out or f'result_logs/ew_probe_{args.calib}_seed{args.seed}'
    runlog.prepare_out_dir(out, 'probe.json')
    runlog.open_mirror(os.path.join(out, 'probe.txt'))

    n_samples_total = args.nsamples * (args.seqlen // args.sample_tokens
                                       if args.sample_tokens else 1)
    log(f'EW-rule プローブ  model={args.model} calib={args.calib} '
        f'seed={args.seed} N={args.nexperts} '
        f'sample_tokens={args.sample_tokens} M={n_samples_total} '
        f'abs={args.use_abs}')

    adapter = create_adapter(guess_adapter(args.model), args.model,
                             seqlen=args.seqlen)
    carve = load_calibration(args.calib, args.model, args.seqlen,
                             args.nsamples, args.seed)
    log(f'  carve: {carve.name} {tuple(carve.input_ids.shape)} '
        f'hash={carve.metadata()["token_hash"][:12]}')

    inputs = adapter.capture_layer_inputs(carve.input_ids)
    adapter.to_device()
    hidden = inputs.hidden
    if args.batch_chunk is None:
        hidden = hidden.to(adapter.device)

    n_layers = args.layers or adapter.n_layers
    cvs = []
    for index in range(n_layers):
        dense = adapter.dense_ffn(index)
        z, residual = adapter.forward_attention(
            index, hidden, inputs.attention_mask, inputs.position_ids,
            batch_chunk=args.batch_chunk)
        profile = layer_gate_profile(dense, z,
                                     sample_tokens=args.sample_tokens,
                                     use_abs=args.use_abs,
                                     batch_chunk=args.batch_chunk,
                                     device=adapter.device)
        cv = neuron_cv(profile)
        cvs.append(cv)
        quantiles = torch.quantile(
            cv, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        log(f'  層 {index:2d}  CV 5/25/50/75/95% = '
            + ' '.join(f'{q:.3f}' for q in quantiles.tolist())
            + f'  max={cv.max():.3f}')
        hidden = dense_ffn_forward(dense, z, residual,
                                   batch_chunk=args.batch_chunk,
                                   device=adapter.device)
        del z, residual, profile

    stacked = torch.stack(cvs)
    np.save(os.path.join(out, 'cv.npy'), stacked.numpy())

    values, detail = ew_allocation(cvs, args.nexperts, args.nactive,
                                   tau=DEFAULT_TAU, alpha_min=DEFAULT_ALPHA_MIN,
                                   alpha_max=DEFAULT_ALPHA_MAX)
    log('')
    log(f'EW 既定値 τ={DEFAULT_TAU} α=[{DEFAULT_ALPHA_MIN}, '
        f'{DEFAULT_ALPHA_MAX}]  A={args.nactive}')
    log('  r_ℓ  ' + ' '.join(f'{r:.3f}' for r in detail['ratios']))
    log('  x_ℓ  ' + ','.join(str(x) for x in values))
    log(f"  平均 x = {detail['mean_x']:.4f}  "
        f"routing のある層 = {detail['n_routing_layers']}/{len(values)}  "
        f"クリップした層 = {detail['n_clipped']}")
    if len(set(values)) == 1:
        log('  ** 縮退している。全層が同じ x で、これは一様配分そのものである。'
            'τ を振り直すまでベンチは回さない **')

    runlog.write_json(os.path.join(out, 'probe.json'), {
        'arguments': vars(args),
        'calibration': carve.metadata(),
        'n_layers': n_layers,
        'default': {'values': values, **detail},
    })
    log(f'\n書いた: {out}')
    runlog.close_mirror()
    return 0


if __name__ == '__main__':
    sys.exit(main())
