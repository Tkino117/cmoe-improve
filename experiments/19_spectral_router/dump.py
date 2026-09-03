"""ルーター探査のための材料を層ごとに書き出す。

``Converter.convert`` と同じ順で層を歩き、指定した層で

* fit / validation の FFN 入力 z（トークンを間引いたもの）
* その層の分割（expert ごとのニューロン添字と代表）
* dense の gate / up / down 重み

を落とす。ルーターの score 関数を試すたびにモデルを走らせ直さないための土台で
あり、これ自体は測定ではない。層に載せて次層へ伝播させるのは常に現行 CMoE の
ルーターなので、落ちる z は report/10・18 と同じ軌道の上にある。

    uv run python experiments/19_router_probe/dump.py --layers 0,7,15,23,31
"""

import argparse
import os
import time

import torch

from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.alloc.search.fixed import parse_allocation
from cmoe.assemble import build_moe
from cmoe.carve.profile import profile_layer
from cmoe.carve.registry import create_carver
from cmoe.data.registry import load_splits
from cmoe.moe.modules import forward_chunked
from cmoe.router.base import build_baseline_router

DEFAULT_OUT = os.path.join(
    os.environ.get('CMOE_PROBE_DIR', '/tmp/claude-1000/probe'), 'dump')


def subsample(z, n_tokens, seed):
    """[bsz, seq, hidden] を [n, hidden] に間引く。"""
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[0] <= n_tokens:
        return flat.clone()
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(flat.shape[0], generator=generator)[:n_tokens]
    return flat[index.to(flat.device)].clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib', default='wikitext2')
    parser.add_argument('--alloc', default='uniform3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nsamples', type=int, default=8)
    parser.add_argument('--fit-samples', type=int, default=64)
    parser.add_argument('--validation-samples', type=int, default=64)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--k-act', type=int, default=10)
    parser.add_argument('--fit-batch-chunk', type=int, default=4)
    parser.add_argument('--layers', default='0,7,15,23,31',
                        help='材料を落とす層')
    parser.add_argument('--tokens', type=int, default=16384,
                        help='fit / validation それぞれから残すトークン数')
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args()

    wanted = {int(field) for field in args.layers.split(',') if field.strip()}
    os.makedirs(args.out, exist_ok=True)

    adapter_name = guess_adapter(args.model)
    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)
    carver = create_carver('cmoe', args.nexperts, k_act=args.k_act)
    splits = load_splits(args.calib, args.model, args.seqlen, args.seed,
                         carve_count=args.nsamples,
                         fit_count=args.fit_samples,
                         validation_count=args.validation_samples)
    carve, fit, validation = splits.carve, splits.fit, splits.validation
    print(f'carve={carve.metadata()["token_hash"][:12]} '
          f'fit={fit.metadata()["token_hash"][:12]} '
          f'val={validation.metadata()["token_hash"][:12]}')

    allocation = parse_allocation(
        args.alloc, n_active_total=args.nactive).search(None, adapter.n_layers)

    carve_inputs = adapter.capture_layer_inputs(carve.input_ids)
    fit_inputs = adapter.capture_layer_inputs(fit.input_ids)
    validation_inputs = adapter.capture_layer_inputs(validation.input_ids)
    adapter.to_device()

    hidden = carve_inputs.hidden.to(adapter.device)
    fit_hidden = fit_inputs.hidden
    validation_hidden = validation_inputs.hidden

    started = time.time()
    for index in range(adapter.n_layers):
        n_shared = allocation[index]
        topk = allocation.topk(index)
        dense = adapter.dense_ffn(index)
        z, residual = adapter.forward_attention(
            index, hidden, carve_inputs.attention_mask,
            carve_inputs.position_ids)
        rates, markers = profile_layer(dense, z, k_act=args.k_act,
                                       normalize=True, device=adapter.device)
        partition = carver.carve(dense, rates, markers, n_shared, z=z)
        router = build_baseline_router(dense, partition, topk)
        moe = build_moe(dense, partition, topk, router, args.nexperts)
        moe.to(adapter.device)

        fit_z, fit_residual = adapter.forward_attention(
            index, fit_hidden, fit_inputs.attention_mask,
            fit_inputs.position_ids, batch_chunk=args.fit_batch_chunk)
        validation_z, validation_residual = adapter.forward_attention(
            index, validation_hidden, validation_inputs.attention_mask,
            validation_inputs.position_ids, batch_chunk=args.fit_batch_chunk)

        if index in wanted:
            payload = {
                'layer': index,
                'n_shared': n_shared,
                'topk': topk,
                'expert_groups': [list(group) for group in partition.expert_groups],
                'representatives': list(partition.representative_indices),
                'rates': rates.detach().to('cpu', torch.float32),
                'gate_weight': dense.gate_proj.weight.detach().to('cpu', torch.float16),
                'up_weight': dense.up_proj.weight.detach().to('cpu', torch.float16),
                'down_weight': dense.down_proj.weight.detach().to('cpu', torch.float16),
                'fit_z': subsample(fit_z, args.tokens, 1000 + index).to(
                    'cpu', torch.float16),
                'validation_z': subsample(validation_z, args.tokens, 2000 + index).to(
                    'cpu', torch.float16),
            }
            path = os.path.join(args.out, f'layer{index:02d}.pt')
            torch.save(payload, path)
            print(f'層 {index:>2}: x={n_shared} Top-K={topk} -> {path} '
                  f'({os.path.getsize(path) / 1e6:.0f} MB)')

        adapter.replace_ffn(index, moe)
        hidden = forward_chunked(moe, z, residual, device=adapter.device)
        fit_hidden = forward_chunked(moe, fit_z, fit_residual,
                                     batch_chunk=args.fit_batch_chunk,
                                     device=adapter.device)
        validation_hidden = forward_chunked(moe, validation_z, validation_residual,
                                            batch_chunk=args.fit_batch_chunk,
                                            device=adapter.device)
        del (dense, z, residual, rates, markers, partition, router, moe,
             fit_z, fit_residual, validation_z, validation_residual)
        torch.cuda.empty_cache()

    print(f'完了 {time.time() - started:.1f}s')


if __name__ == '__main__':
    main()
