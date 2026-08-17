"""活性プロファイル: どのニューロンがどれだけの頻度で上位に入るか。

CMoE-ref の ``CMoE_utils.analyze_neuron_activations`` の移送。トークンごとに
|h| の上位 K 個へ印を付け、その印の集計が activation rate になる。

トークンごとの topk ループはそのまま残してある。全体を一度に ``torch.topk``
すれば速いが、同点の解け方が変わりうる。ここは既存の測定と一致させることが
先で、高速化は数値が合ってからでよい。
"""

import torch


@torch.no_grad()
def analyze_activations(scores, k_act=10):
    """[batch, seq, inter] の h から (counts, rates, markers) を返す。

    markers は [batch*seq, inter] の 0/1。rates は markers の列平均。
    """
    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len

    flat_states = scores.reshape(-1, inter_size)
    markers = torch.zeros_like(flat_states)

    for i in range(total_samples):
        abs_values = flat_states[i].abs().float()
        _, top_indices = torch.topk(abs_values, k=k_act)
        markers[i, top_indices] = 1.0

    counts = markers.sum(dim=0)
    rates = counts / total_samples
    return counts, rates, markers


@torch.no_grad()
def hidden_activations(dense, z, normalize=True):
    """FFN 入力 z から中間活性 h を作る。

    normalize=True（既定）は CMoE のプロファイリング用 h で、入力と重みを
    L2 正規化してから掛ける。False は素の h（= 真の中間活性 H）。

    CMoE-ref では前者が ``construct_moe`` の既定枝、後者が
    ``--no-profiling-norm`` 枝にあたる。
    """
    import torch.nn.functional as F

    if normalize:
        z_n = F.normalize(z, p=2, dim=-1)
        h = dense.act_fn(F.linear(z_n, F.normalize(dense.gate_proj.weight, p=2, dim=1)))
        return h * F.linear(z_n, F.normalize(dense.up_proj.weight, p=2, dim=1))
    h = dense.act_fn(F.linear(z, dense.gate_proj.weight))
    return h * F.linear(z, dense.up_proj.weight)
