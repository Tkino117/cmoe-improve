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
def select_positions(z, mask):
    """層の FFN 入力から、印の付いた位置だけを取り出す。

    ``[bsz, seq, hidden]`` を ``[1, 印の数, hidden]`` にする。プロファイルは
    ``[トークン, ニューロン]`` に潰してから数えるので、系列の形は結果に効か
    ない。1本にまとめるので ``batch_chunk`` はここでは効かなくなるが、確保の
    上限は印の数（校正セットが決める採点位置の総量）で抑えられており、絞る
    前の全位置を数えていたときより小さい。

    ``mask`` が None なら z をそのまま返す（既存のすべての経路がこれで、値も
    確保も1ビットも変わらない）。

    **絞るのはプロファイルだけである。** 次の層へ渡す出力も、層ローカル指標が
    読む真の H も、絞っていない z から作る。印が言うのは「どの位置の活性が
    採点に効くか」であって、「どの位置を走らせるか」ではない。
    """
    if mask is None:
        return z
    if z.dim() != 3:
        raise ValueError(f'[bsz, seq, hidden] のはず（{tuple(z.shape)}）')
    if tuple(mask.shape) != tuple(z.shape[:2]):
        raise ValueError(
            f'印は {tuple(mask.shape)}、z は {tuple(z.shape[:2])} — '
            '校正トークンとこの層の入力が対応していない')
    flat = z.reshape(-1, z.shape[-1])[mask.reshape(-1).to(z.device)]
    if flat.shape[0] == 0:
        raise ValueError('印の付いた位置が1つも無い')
    return flat.unsqueeze(0)


@torch.no_grad()
def profile_layer(dense, z, k_act=10, normalize=True, batch_chunk=None,
                  device=None):
    """FFN 入力 z から (rates, markers) を作る。バッチを分けても結果は同じ。

    組み立て役と配分オラクルが同じ統計を見るための1本。分割規則は統計から
    決まるので、ここが2実装あると「探索が測った分割」と「変換が載せた分割」が
    別物になりうる。
    """
    step = batch_chunk or z.shape[0]
    rows = []
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step].to(device) if device is not None else z[start:start + step]
        h = hidden_activations(dense, chunk, normalize=normalize)
        rows.append(h.to('cpu'))
        del h, chunk
    h = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
    _, rates, markers = analyze_activations(h, k_act=k_act)
    return rates, markers


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


@torch.no_grad()
def marker_weights(dense, z, k_act=10, normalize=True, batch_chunk=None,
                   device=None):
    """印の付いた場所に ``|h|`` を置いた [トークン, ニューロン]。

    ``analyze_activations`` が返す markers は 0/1 で、上位10に入ったかどうか
    しか残らない。取りこぼす量は活性の**大きさ**で決まるのに、分割規則は
    その大きさを見ずにクラスタを作っている。その差を埋めたい方式がこれを使う。

    印の位置は markers と同じ（同じ k_act の topk）で、値だけが違う。
    """
    step = batch_chunk or z.shape[0]
    rows = []
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step].to(device) if device is not None else z[start:start + step]
        h = hidden_activations(dense, chunk, normalize=normalize)
        rows.append(h.to('cpu'))
        del h, chunk
    h = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]

    flat = h.reshape(-1, h.shape[-1]).abs().float()
    weights = torch.zeros_like(flat)
    for index in range(flat.shape[0]):
        values, top_indices = torch.topk(flat[index], k=k_act)
        weights[index, top_indices] = values
    return weights


@torch.no_grad()
def neuron_mass(dense, z, batch_chunk=None, device=None):
    """ニューロンごとの ``Σ_t |H_ti|``（真の中間活性、fp32）。

    ``alloc.oracles.mass`` が回収率を測るときの質量と同じものを、ニューロン
    単位で足したものである。プロファイル用の正規化した h ではなく**真の H**
    から取る、という ``mass`` 側の規則をここでも守る。
    """
    step = batch_chunk or z.shape[0]
    total = None
    for start in range(0, z.shape[0], step):
        chunk = z[start:start + step].to(device) if device is not None else z[start:start + step]
        h = hidden_activations(dense, chunk, normalize=False)
        part = h.reshape(-1, h.shape[-1]).abs().to(torch.float32).sum(dim=0).to('cpu')
        total = part if total is None else total + part
        del h, chunk, part
    return total
