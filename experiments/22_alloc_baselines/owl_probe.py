"""22-a OWL の層別統計（外れ値比率 D_ℓ）を dense の軌道の上で測る。

**ベンチを回す前に必ずここを通す。** EW-rule の CV と同じ落とし穴がある —
D_ℓ が全層でほぼ同じなら、どんな α レンジを与えても規則は一様配分に潰れ、
数時間のベンチで分かるのは「一様配分をもう一度測った」ことだけになる。
``allocate.py`` が ``degenerate`` を見て止める材料をここが作る。

D_ℓ は M（外れ値のしきい値の倍率）にしか依存せず、N・A・α のどれにも依存しない。
だからこの1回の前向きから、M の格子もスパース率の違いも CPU だけで出せる。

  uv run python experiments/22_alloc_baselines/owl_probe.py --seed 0
  uv run python experiments/22_alloc_baselines/owl_probe.py --seed 0 --layers 2

校正・動作点は report/09・16・21 と揃えてある（slimpajama n=16 × 2048、N=8）。
**dense の軌道の上で測る**のは OWL が元の密なモデルの重みと活性を読むからで、
変換済みの接頭辞の上で測るのは別の量である（EW-rule の probe と同じ理由）。
"""

import argparse
import os
import sys

import torch

from cmoe import runlog
from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.alloc.owl import DEFAULT_M, M_GRID, ratios_from_norms
from cmoe.carve.profile import hidden_activations
from cmoe.data.registry import load_calibration

log = runlog.log


@torch.no_grad()
def dense_step(dense, z, residual, batch_chunk=None, device=None):
    """dense の FFN を通した次の層の入力と、‖X_j‖₂ 2本を1回の走査で作る。

    z と H を両方持ち回らないために、norm を塊ごとに積みながら進める。7B の
    1層で H は [32768, 11008] あり、fp32 に上げると 1.4 GiB になる。
    """
    step = batch_chunk or z.shape[0]
    sum_z = torch.zeros(z.shape[-1], dtype=torch.float64)
    sum_h = None
    rows = []
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step]
        base = residual[start:start + step]
        if device is not None:
            chunk, base = chunk.to(device), base.to(device)
        flat = chunk.reshape(-1, chunk.shape[-1]).to(torch.float32)
        sum_z += flat.square().sum(dim=0).double().cpu()
        del flat

        h = hidden_activations(dense, chunk, normalize=False)
        flat_h = h.reshape(-1, h.shape[-1]).to(torch.float32)
        if sum_h is None:
            sum_h = torch.zeros(flat_h.shape[-1], dtype=torch.float64)
        sum_h += flat_h.square().sum(dim=0).double().cpu()
        del flat_h

        rows.append((base + dense.down_proj(h)).to(residual.device))
        del h, chunk, base
    return (torch.cat(rows, dim=0),
            sum_z.sqrt().to(torch.float32), sum_h.sqrt().to(torch.float32))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='owl-probe', description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=16)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--batch-chunk', type=int, default=4,
                        help='1回の前向きで進める系列数。H を fp32 に上げるので'
                             '既定で分けてある')
    parser.add_argument('--layers', type=int, default=None,
                        help='先頭 n 層だけ。動作確認用')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    out = args.out or f'result_logs/owl_probe_{args.calib}_seed{args.seed}'
    runlog.prepare_out_dir(out, 'owl.json')
    runlog.open_mirror(os.path.join(out, 'owl.txt'))

    log(f'OWL プローブ  model={args.model} calib={args.calib} '
        f'seed={args.seed} n={args.nsamples} M格子={list(M_GRID)}')
    log('  統計は FFN の3枚（gate / up / down）だけで数える。'
        'この土俵で動かすのは FFN の配分なので、attention を混ぜると'
        '配分の決め方とは別の情報が入る（原典から違えた点）')

    adapter = create_adapter(guess_adapter(args.model), args.model,
                             seqlen=args.seqlen)
    carve = load_calibration(args.calib, args.model, args.seqlen,
                             args.nsamples, args.seed)
    log(f'  carve: {carve.name} {tuple(carve.input_ids.shape)} '
        f'hash={carve.metadata()["token_hash"][:12]}')

    inputs = adapter.capture_layer_inputs(carve.input_ids)
    adapter.to_device()
    hidden = inputs.hidden

    n_layers = min(args.layers or adapter.n_layers, adapter.n_layers)
    rows = []
    for index in range(n_layers):
        dense = adapter.dense_ffn(index)
        z, residual = adapter.forward_attention(
            index, hidden, inputs.attention_mask, inputs.position_ids,
            batch_chunk=args.batch_chunk)
        hidden, norm_z, norm_h = dense_step(
            dense, z, residual, batch_chunk=args.batch_chunk,
            device=adapter.device)
        ratios, detail = ratios_from_norms(dense, norm_z, norm_h,
                                           m_grid=M_GRID,
                                           device=adapter.device)
        rows.append({'layer': index, 'ratios': ratios, **detail})
        log(f'  層 {index:2d}  D = '
            + '  '.join(f'M={m:g}:{ratios[m]:.5f}' for m in M_GRID))
        del z, residual

    log('')
    for m in M_GRID:
        values = [row['ratios'][m] for row in rows]
        spread = max(values) - min(values)
        mark = '  ** ほぼ一定。規則は潰れる **' if spread < 1e-6 else ''
        log(f'M={m:g}  D_ℓ  min={min(values):.5f} max={max(values):.5f} '
            f'幅={spread:.5f}{mark}')
    log(f'\n既定は M={DEFAULT_M:g}。α への写像と向きは allocate.py が決める')

    runlog.write_json(os.path.join(out, 'owl.json'), {
        'arguments': vars(args),
        'calibration': carve.metadata(),
        'n_layers': n_layers,
        'm_grid': list(M_GRID),
        'default_m': DEFAULT_M,
        'layers': rows,
    })
    log(f'\n書いた: {out}')
    runlog.close_mirror()
    return 0


if __name__ == '__main__':
    sys.exit(main())
