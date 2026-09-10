"""28-1 LLaMA-MoE-v2 の分割に要る「層ごと・クラスタごとの重要度」を作る。

移送元は `OpenSparseLLMs/LLaMA-MoE-v2` の
`smoe/entrypoint/expert_construction/get_gates/hidden_feature_clustering.py`（ゲート）と
`smoe/entrypoint/expert_construction/split/split_gradient_get_grads_v2.py`（重要度）。

2段。

1. **ゲート** … 層ごとに MLP 入力の隠れ状態を、routed expert と同数のクラスタへ
   balanced k-means で分ける。中心が原典の gate 重みになる
2. **重要度** … LM 損失を backward し、各トークンを ``argmax(h · center^T)`` で
   クラスタへ割り当て、中間活性 f について ``|f ⊙ ∇_f L|`` をクラスタ内で平均する

**原典との違い。**

* クラスタリングは原典が `k_means_constrained.KMeansConstrained`（サイズ制約
  つき、jitter 0.4）を使う。依存を増やさずに済ませるため、ここは Lloyd 法の
  割り当て段だけを**同じ下限・上限**で貪欲に解く。厳密解ではないので、
  「サイズ制約つき k-means の近似」と読むこと
* 校正は提案手法と同じ slimpajama（原典は SFT データ）
* **追加学習はしない**

クラスタ数は routed expert 数なので、**x を変えたら取り直す**。

  uv run python experiments/28_llama_moe/probe.py --seed 0 --nactive 6 --nshared 1
"""

import argparse
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from cmoe import runlog                                        # noqa: E402
from cmoe.adapters.registry import create_adapter, guess_adapter  # noqa: E402
from cmoe.data.registry import load_calibration                # noqa: E402

# 原典の balance_jitter_factor。クラスタサイズが均等から何割ずれてよいか
JITTER = 0.4


def balanced_kmeans(features, n_clusters, jitter=JITTER, iters=10, seed=0):
    """サイズ制約つき k-means の近似。features は [トークン, hidden]。

    割り当ては「最良と次点の差が大きい点（迷いの少ない点）から席を取る」貪欲法。
    下限・上限は原典と同じ ``floor(n/k·(1-jitter))`` / ``ceil(n/k·(1+jitter))``
    で、下限を割ったクラスタには、余っているクラスタから近い点をまとめて移す。
    厳密解ではないが、目的（トークンを機能で束ねる）には足りる。
    """
    n_points = features.shape[0]
    if n_clusters < 1 or n_points < n_clusters:
        raise ValueError(f'{n_points} 点を {n_clusters} クラスタには割れない')
    balanced = n_points / n_clusters
    low = max(0, math.floor(balanced * (1 - jitter)))
    high = min(n_points, math.ceil(balanced * (1 + jitter)))

    generator = torch.Generator(device='cpu').manual_seed(seed)
    start = torch.randperm(n_points, generator=generator)[:n_clusters]
    centres = features[start].clone()
    for _ in range(iters):
        distance = torch.cdist(features, centres)          # [点, クラスタ]
        order = torch.argsort(distance, dim=1)
        best = distance.gather(1, order[:, :1]).squeeze(1)
        second = (distance.gather(1, order[:, 1:2]).squeeze(1)
                  if n_clusters > 1 else best)
        priority = torch.argsort(second - best, descending=True).tolist()
        choices = order.tolist()
        labels = [0] * n_points
        counts = [0] * n_clusters
        for point in priority:
            for choice in choices[point]:
                if counts[choice] < high:
                    labels[point] = choice
                    counts[choice] += 1
                    break
        labels = torch.tensor(labels, device=features.device)
        labels = _fill_up_to_low(labels, counts, distance, low, n_clusters)
        new_centres = torch.stack([
            features[labels == cluster].mean(dim=0)
            if bool((labels == cluster).any()) else centres[cluster]
            for cluster in range(n_clusters)])
        if torch.allclose(new_centres, centres):
            centres = new_centres
            break
        centres = new_centres
    return centres


def _fill_up_to_low(labels, counts, distance, low, n_clusters):
    """下限を割ったクラスタへ、余っているクラスタから近い点をまとめて移す。

    1点ずつ動かすと「点数 × 移動回数」になって（16k 点で数千万回）終わらないので、
    クラスタごとに必要数を一度に選ぶ。近似が一段深くなるが、これは balanced
    k-means の近似そのものであり、クラスタ中心の質にしか効かない。
    """
    for cluster in range(n_clusters):
        deficit = low - counts[cluster]
        if deficit <= 0:
            continue
        spare = torch.tensor([counts[other] > low for other in range(n_clusters)],
                             device=labels.device)
        movable = spare[labels] & (labels != cluster)
        candidates = movable.nonzero().view(-1)
        if candidates.numel() == 0:
            continue
        take = candidates[torch.argsort(distance[candidates, cluster])[:deficit]]
        for point in take.tolist():
            counts[int(labels[point])] -= 1
        labels[take] = cluster
        counts[cluster] += int(take.numel())
    return labels


@torch.no_grad()
def gate_weights(adapter, calibration, n_clusters, batch_chunk, seed, log,
                 gate_tokens=None):
    """層ごとのクラスタ中心。原典の gate 重みに当たる。

    ``gate_tokens`` を渡すとクラスタリングに使うトークンをそこまで間引く。
    割り当ての貪欲法が Python のループなので、32k トークンを全部使うと層あたり
    分単位になる。**中心の推定に要るのは点の分布であって全点ではない**ので、
    種を固定した一様抜き取りで足りる（重要度の側は全トークンを使う）。
    """
    inputs = adapter.capture_layer_inputs(calibration.input_ids)
    hidden = inputs.hidden.to(adapter.device)
    centres = []
    for index in range(adapter.n_layers):
        z, _ = adapter.forward_attention(
            index, hidden, inputs.attention_mask, inputs.position_ids,
            batch_chunk=batch_chunk)
        features = z.reshape(-1, z.shape[-1]).to(torch.float32)
        if gate_tokens and features.shape[0] > gate_tokens:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            keep = torch.randperm(features.shape[0], generator=generator)[:gate_tokens]
            features = features[keep.to(features.device)]
        centres.append(balanced_kmeans(features, n_clusters, seed=seed).cpu())
        hidden = adapter.forward_layer(
            index, hidden, inputs.attention_mask, inputs.position_ids)
        log(f'  層 {index:>2} ゲート {tuple(centres[-1].shape)}')
    del hidden
    torch.cuda.empty_cache()
    return centres


def importance(adapter, calibration, centres, n_clusters, token_chunk, log):
    """``|f ⊙ ∇_f L|`` をクラスタごとに平均する。

    ``f`` は down_proj の入力（= silu(gate)·up）で、原典が重要度を測る場所と
    同じである。トークンのクラスタは up_proj の入力（= MLP 入力）で決まる。
    """
    model = adapter.model
    device = adapter.device
    layers = adapter.layers
    n_layers = len(layers)
    intermediate = model.config.intermediate_size

    totals = [torch.zeros(n_clusters, intermediate, dtype=torch.float64)
              for _ in range(n_layers)]
    counts = [torch.zeros(n_clusters, dtype=torch.float64)
              for _ in range(n_layers)]
    cluster_ids, features = {}, {}
    handles = []

    def on_mlp_input(index):
        def hook(_module, inp, _out):
            hidden = inp[0].detach().reshape(-1, inp[0].shape[-1])
            logits = hidden.to(torch.float32) @ centres[index].to(device).t()
            cluster_ids[index] = torch.argmax(logits, dim=-1)
        return hook

    def on_intermediate(index):
        def hook(_module, inp, _out):
            features[index] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        return hook

    def on_intermediate_grad(index):
        def hook(_module, grad_in, _grad_out):
            grad = grad_in[0]
            if grad is None:
                return
            grad = grad.detach().reshape(-1, grad.shape[-1]).to(torch.float32)
            values = (grad * features[index].to(torch.float32)).abs()
            ids = cluster_ids[index]
            for cluster in range(n_clusters):
                mask = ids == cluster
                if not bool(mask.any()):
                    continue
                totals[index][cluster] += values[mask].sum(dim=0).double().cpu()
                counts[index][cluster] += float(mask.sum())
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.mlp.up_proj.register_forward_hook(on_mlp_input(index)))
        handles.append(layer.mlp.down_proj.register_forward_hook(on_intermediate(index)))
        handles.append(layer.mlp.down_proj.register_full_backward_hook(
            on_intermediate_grad(index)))

    for param in model.parameters():
        param.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    ids = calibration.input_ids.to(device)
    total, seqlen = ids.shape
    step = max(1, token_chunk // seqlen)
    try:
        for start in range(0, total, step):
            chunk = ids[start:start + step]
            loss = model(chunk, labels=chunk).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            log(f'  backward {start + chunk.shape[0]}/{total} '
                f'loss={float(loss.detach()):.4f}')
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.requires_grad_(False)
        torch.cuda.empty_cache()

    scores = []
    for index in range(n_layers):
        divisor = counts[index].clamp(min=1).unsqueeze(1)
        scores.append((totals[index] / divisor).float())
        empty = int((counts[index] == 0).sum())
        if empty:
            log(f'  注意: 層 {index} で {empty} クラスタにトークンが入らなかった')
    return scores, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--calib', default='slimpajama')
    parser.add_argument('--nsamples', type=int, default=16)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nexperts', type=int, default=8)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--nshared', type=int, default=1,
                        help='residual expert の数。クラスタ数 = nexperts - これ')
    parser.add_argument('--batch-chunk', type=int, default=4)
    parser.add_argument('--token-chunk', type=int, default=2048,
                        help='1回の backward に載せるトークン数の上限')
    parser.add_argument('--gate-tokens', type=int, default=4096,
                        help='ゲートのクラスタリングに使うトークン数の上限')
    parser.add_argument('--layers', type=int, default=None,
                        help='先頭 N 層だけ（動作確認用）')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out = args.out or runlog.default_out_dir('llama_moe_v2_probe')
    runlog.prepare_out_dir(out, 'scores.json')
    runlog.open_mirror(os.path.join(out, 'run.txt'))
    log = runlog.log

    n_clusters = args.nexperts - args.nshared
    log(f'model={args.model} calib={args.calib} n={args.nsamples} seed={args.seed}')
    log(f'N={args.nexperts} A={args.nactive} residual={args.nshared} '
        f'クラスタ={n_clusters}')
    log(f'出力 {out}')

    adapter_name = args.adapter or guess_adapter(args.model)
    adapter = create_adapter(adapter_name, args.model, seqlen=args.seqlen)
    adapter.to_device()
    if args.layers:
        adapter.model.model.layers = adapter.layers[:args.layers]
    calibration = load_calibration(
        args.calib, args.model, args.seqlen, args.nsamples, args.seed)
    log(f'calib: {calibration.name} {tuple(calibration.input_ids.shape)} '
        f'hash={calibration.metadata()["token_hash"][:12]}')

    started = time.time()
    log('== ゲート（層ごとの balanced k-means）==')
    centres = gate_weights(adapter, calibration, n_clusters, args.batch_chunk,
                           args.seed, log, gate_tokens=args.gate_tokens)
    log(f'  {time.time() - started:.1f}s')

    log('== 重要度（|f ⊙ ∇_f L|）==')
    started_scores = time.time()
    scores, counts = importance(adapter, calibration, centres, n_clusters,
                                args.token_chunk, log)
    log(f'  {time.time() - started_scores:.1f}s')

    payload = {
        'arguments': vars(args),
        'n_clusters': n_clusters,
        'data': {'calibration': calibration.metadata()},
        'token_counts': [row.tolist() for row in counts],
        'scores': [row.tolist() for row in scores],
        'seconds': time.time() - started,
    }
    runlog.write_json(os.path.join(out, 'scores.json'), payload)
    log(f'{len(scores)} 層ぶんを書いた')
    runlog.close_mirror()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
