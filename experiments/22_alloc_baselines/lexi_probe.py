"""22-b LExI の感度プロファイル（層ローカルの出力誤差の表）を dense の軌道で作る。

LExI は「層 ℓ の active expert 数だけを動かし、**他の層は baseline のまま**」
にして、その層の出力のずれを測る。ここが作るのはその表 ``L[ℓ][x]`` で、
``allocate.py`` が層ごとの argmin を取ると LExI の配分になる（予算制約がこの
土俵では消えるので、原典の進化探索は argmin に厳密に退化する — 理由は
``cmoe.alloc.lexi`` の冒頭に書いた）。

**探索の ``greedy`` とは別物である。** あちらは変換済みの接頭辞の上を進むので、
層 ℓ の候補は層 0..ℓ−1 で選んだ x に依存する。ここは全層 dense のまま進めるので、
どの層の表も他の層の選択に依存しない。その差が「層ローカルか、後続まで通すか」
の差であり、この対照を置く理由そのものである。

表は A に依存する（候補 x の上限が A で、Top-K = A − x が L に効く）ので、
スパース率ごとに1回ずつ走らせる。M や τ のようなしきい値は無い。

  uv run python experiments/22_alloc_baselines/lexi_probe.py --seed 0 --nactive 6
  uv run python experiments/22_alloc_baselines/lexi_probe.py --seed 0 --nactive 4
  uv run python experiments/22_alloc_baselines/lexi_probe.py --seed 0 --layers 2

校正・動作点・分割・ルーターは report/09・16・21 と同一である。
"""

import argparse
import os
import sys
import time

import torch

from cmoe import runlog
from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.alloc.lexi import argmin_allocation
from cmoe.alloc.oracles.base import LayerWalk, PrefixState
from cmoe.alloc.oracles.local_error import LocalErrorOracle
from cmoe.assemble import layer_factory
from cmoe.carve.registry import create_carver
from cmoe.data.registry import load_calibration

log = runlog.log


@torch.no_grad()
def dense_forward(dense, z, residual, batch_chunk=None, device=None):
    """dense の FFN を通した次の層の入力。baseline の軌道を作る役。"""
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
    parser = argparse.ArgumentParser(prog='lexi-probe', description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=16)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--carver', default='cmoe')
    parser.add_argument('--k-act', type=int, default=10)
    parser.add_argument('--bias-speed', type=float, default=0.001)
    parser.add_argument('--batch-chunk', type=int, default=None)
    parser.add_argument('--token-chunk', type=int, default=None)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    out = args.out or f'result_logs/lexi_probe_{args.calib}{tag}_seed{args.seed}'
    runlog.prepare_out_dir(out, 'lexi.json')
    runlog.open_mirror(os.path.join(out, 'lexi.txt'))

    log(f'LExI 感度プローブ  model={args.model} calib={args.calib} '
        f'seed={args.seed} N={args.nexperts} A={args.nactive}')
    log('  他の層は dense のまま（baseline）。層ごとの表は互いに独立になる')

    adapter = create_adapter(guess_adapter(args.model), args.model,
                             seqlen=args.seqlen)
    calibration = load_calibration(args.calib, args.model, args.seqlen,
                                   args.nsamples, args.seed)
    log(f'  carve: {calibration.name} {tuple(calibration.input_ids.shape)} '
        f'hash={calibration.metadata()["token_hash"][:12]}')

    inputs = adapter.capture_layer_inputs(calibration.input_ids)
    factory = layer_factory(
        create_carver(args.carver, args.nexperts, k_act=args.k_act),
        args.nexperts, bias_speed=args.bias_speed, router_norm=True,
        device=adapter.device)
    walk = LayerWalk(adapter, inputs, factory, args.nexperts,
                     n_active_total=args.nactive, k_act=args.k_act,
                     profiling_norm=True, batch_chunk=args.batch_chunk)
    oracle = LocalErrorOracle(walk)
    if args.token_chunk is not None:
        oracle.token_chunk = args.token_chunk

    adapter.to_device()
    hidden = inputs.hidden.to(walk.state_device)
    n_layers = min(args.layers or adapter.n_layers, adapter.n_layers)

    started = time.time()
    table = []
    for index in range(n_layers):
        # pinned=True にしておくのは、この状態の hidden をこちらが持ち回るから
        # である（release で None にされると次の層の入力が消える）
        state = PrefixState(hidden=hidden, depth=index, score=0.0, pinned=True)
        profile = walk.profile(state, index)
        row = {}
        for x in walk.candidates(index):
            carved = walk.carve(profile, x)
            result = oracle.measure(profile, carved, state, None)
            row[x] = result.details['l']
            if result.details['over_unity']:
                log(f'  ** 層 {index} x={x} で L>1（打ち消しが起きている）')
            del carved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        table.append(row)
        best = min(row, key=row.get)
        log(f'  層 {index:2d}  ' + '  '.join(f'x={x}:{row[x]:.4e}'
                                             for x in sorted(row))
            + f'   argmin x={best}')
        hidden = dense_forward(profile.dense, profile.z, profile.residual,
                               batch_chunk=args.batch_chunk,
                               device=adapter.device).to(walk.state_device)
        walk.release(state)
        del profile, state

    values, detail = argmin_allocation(table, args.nactive)
    log('')
    log(f'LExI 相当の配分（層ごとの独立な argmin、同点は x の小さい方）')
    log('  ' + ','.join(str(x) for x in values))
    log(f"  平均x={detail['mean_x']:.4f}  "
        f"routing のある層={detail['n_routing_layers']}/{n_layers}  "
        f"Top-K=0 の層={detail['n_no_routing_layers']}/{n_layers}")
    if detail['degenerate']:
        log('  ** 全層同じ x。この対照は一様配分に潰れている **')

    runlog.write_json(os.path.join(out, 'lexi.json'), {
        'arguments': vars(args),
        'calibration': calibration.metadata(),
        'n_layers': n_layers,
        'oracle': {'name': oracle.name, 'cost_unit': oracle.cost_unit},
        'spent': oracle.spent,
        'calls': oracle.calls,
        'seconds': time.time() - started,
        'allocation': {'values': values, **detail},
    })
    log(f'\n所要 {(time.time() - started) / 60:.1f}分  書いた: {out}')
    runlog.close_mirror()
    return 0


if __name__ == '__main__':
    sys.exit(main())
