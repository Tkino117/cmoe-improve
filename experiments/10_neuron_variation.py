"""10 ニューロンの直流成分と変動を測る（診断のみ）。

report/11 のあと、分割規則の shared を質量 ``Σ_t |H|`` で切る方式
（``cmoe_mass``）を測った。層の出力誤差は 14.7% 減ったのに、選択問題の正答率は
0.008 下がった。説明の候補は「質量で選ぶと、大きいがトークンによって動かない
ニューロンが常時オンの枠を占め、識別に使える変動が枠から押し出される」である。

ここはその**前提だけ**を確かめる。変換もしないし PPL もベンチも測らない。
層ごとにニューロン 11,008 個の統計を出して、

* 「大きいが動かない」ニューロンが実在するか（``|m| / mean|H|`` の分布）
* 基準（現行の頻度 / 質量 / 変動）で shared 枠がどれだけ入れ替わるか
* 質量基準だけが拾うニューロンが、本当に定数寄りか

を見る。前提が立たなければ、説明の方を作り直す。

  uv run python experiments/10_neuron_variation.py
  uv run python experiments/10_neuron_variation.py --layers 4   # 配線確認

真の H（素の重み × 素の入力）から取る。プロファイル用の正規化した h ではない
— ``alloc.oracles.mass`` が質量を測るときの規則に合わせる。
"""

import argparse
import json
from pathlib import Path

import torch

from cmoe.adapters.registry import create_adapter, guess_adapter
from cmoe.carve.profile import hidden_activations, profile_layer
from cmoe.data.registry import load_calibration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result_logs' / 'neuron_stats_benchtrain_seed0.json'


@torch.no_grad()
def layer_statistics(dense, z, k_act, batch_chunk=None, device=None):
    """1層分。ニューロンごとの平均・質量・標準偏差と、現行基準の頻度。"""
    step = batch_chunk or z.shape[0]
    total = squares = absolute = None
    n_tokens = 0
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step]
        chunk = chunk.to(device) if device is not None else chunk
        h = hidden_activations(dense, chunk, normalize=False)
        flat = h.reshape(-1, h.shape[-1]).to(torch.float32)
        part = (flat.sum(dim=0), (flat * flat).sum(dim=0), flat.abs().sum(dim=0))
        total = part[0] if total is None else total + part[0]
        squares = part[1] if squares is None else squares + part[1]
        absolute = part[2] if absolute is None else absolute + part[2]
        n_tokens += flat.shape[0]
        del h, flat, chunk, part

    mean = total / n_tokens
    mass = absolute / n_tokens
    variance = torch.clamp(squares / n_tokens - mean * mean, min=0.0)
    rates, _ = profile_layer(dense, z, k_act=k_act, batch_chunk=batch_chunk,
                             device=device)
    return {
        'mean': mean.cpu(),
        'mass': mass.cpu(),
        'std': variance.sqrt().cpu(),
        'rate': rates.cpu().to(torch.float32),
        'w_norm': dense.down_proj.weight.to(torch.float32).norm(dim=0).cpu(),
        'n_tokens': n_tokens,
    }


def shared_set(scores, size):
    return set(torch.topk(scores, size)[1].tolist())


def summarize(stats, n_experts, n_shared):
    """基準ごとの shared 枠の重なりと、質量基準だけが拾う群の性質。"""
    inter = stats['mean'].shape[0]
    size = (inter // n_experts) * n_shared
    dc_ratio = (stats['mean'].abs() / stats['mass'].clamp(min=1e-12))

    picks = {name: shared_set(stats[name], size)
             for name in ('rate', 'mass', 'std')}
    only_mass = picks['mass'] - picks['rate']
    only_std = picks['std'] - picks['rate']
    index = torch.tensor(sorted(only_mass), dtype=torch.long)
    rest = torch.tensor(sorted(set(range(inter)) - picks['mass']),
                        dtype=torch.long)
    return {
        'shared_size': size,
        # 直流成分が質量のどれだけを占めるか。1 に近いほど「動かない」
        'dc_ratio_mean': float(dc_ratio.mean()),
        'dc_ratio_p50': float(dc_ratio.median()),
        'dc_ratio_p90': float(dc_ratio.quantile(0.9)),
        'n_dc_over_0.9': int((dc_ratio > 0.9).sum()),
        'n_dc_over_0.99': int((dc_ratio > 0.99).sum()),
        # 基準どうしの重なり
        'overlap_rate_mass': len(picks['rate'] & picks['mass']) / size,
        'overlap_rate_std': len(picks['rate'] & picks['std']) / size,
        'overlap_mass_std': len(picks['mass'] & picks['std']) / size,
        'n_only_mass': len(only_mass),
        'n_only_std': len(only_std),
        # 質量基準だけが拾う群は定数寄りか
        'dc_ratio_only_mass': float(dc_ratio[index].mean()) if len(index) else None,
        'dc_ratio_not_shared_by_mass': float(dc_ratio[rest].mean()),
        'std_only_mass': float(stats['std'][index].mean()) if len(index) else None,
        'std_all': float(stats['std'].mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib', default='benchtrain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=8)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nshared', type=int, default=4, help='x。既定は 4')
    parser.add_argument('--k-act', type=int, default=10)
    parser.add_argument('--batch-chunk', type=int, default=2)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)

    adapter = create_adapter(guess_adapter(args.model), args.model,
                             seqlen=args.seqlen)
    calibration = load_calibration(args.calib, args.model, args.seqlen,
                                   args.nsamples, args.seed)
    print(f'{calibration.name} {tuple(calibration.input_ids.shape)} '
          f'hash={calibration.metadata()["token_hash"][:12]}')

    inputs = adapter.capture_layer_inputs(calibration.input_ids)
    hidden = inputs.hidden
    n_layers = min(args.layers or adapter.n_layers, adapter.n_layers)

    rows = []
    print(f'{"層":>3} {"直流比 中央値":>12} {">0.9":>6} {">0.99":>6} '
          f'{"頻度∩質量":>10} {"頻度∩変動":>10} {"質量∩変動":>10} '
          f'{"質量のみ群の直流比":>18}')
    for index in range(n_layers):
        dense = adapter.dense_ffn(index)
        z, residual = adapter.forward_attention(
            index, hidden, inputs.attention_mask, inputs.position_ids,
            batch_chunk=args.batch_chunk)
        stats = layer_statistics(dense, z, args.k_act,
                                 batch_chunk=args.batch_chunk,
                                 device=adapter.device)
        row = summarize(stats, args.nexperts, args.nshared)
        row['layer'] = index
        rows.append(row)
        print(f'{index:>3} {row["dc_ratio_p50"]:>12.4f} '
              f'{row["n_dc_over_0.9"]:>6} {row["n_dc_over_0.99"]:>6} '
              f'{row["overlap_rate_mass"]:>10.3f} {row["overlap_rate_std"]:>10.3f} '
              f'{row["overlap_mass_std"]:>10.3f} '
              f'{row["dc_ratio_only_mass"]:>18.4f}')

        # 次の層の入力。dense のまま進める（変換はしない）。DenseFFN は
        # 重みを読むための入れ物なので、FFN の計算はここで書き下す
        hidden = torch.empty_like(residual)
        step = args.batch_chunk or z.shape[0]
        for start in range(0, z.shape[0], step):
            chunk = z[start:start + step].to(adapter.device)
            out = dense.down_proj(
                dense.act_fn(dense.gate_proj(chunk)) * dense.up_proj(chunk))
            hidden[start:start + step] = (
                residual[start:start + step].to(adapter.device) + out
            ).to(residual.device)
            del chunk, out
        del z, residual, stats, dense
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        'model': args.model,
        'calibration': calibration.metadata(),
        'arguments': vars(args),
        'layers': rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(f'\n{path} に書いた')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
