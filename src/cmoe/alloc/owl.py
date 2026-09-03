"""OWL (Outlier Weighed Layerwise sparsity, ICML 2024) の層別統計を、この土俵に移す。

**提案手法ではない。対照である。** OWL 全体（非構造プルーニング + Wanda/SparseGPT
での枝刈り）の再現ではなく、**層ごとのスカラーを作る式だけ**を取り出して、
``alloc.score_rule`` の写像に載せる。比べているのは配分軸だけである。

移す理由は EW-rule と同じで、両者が「層の活性統計 → 層別スカラー → 単調写像 →
層別の予算」という同じ形をしているからである。**規則型の対照が EW-rule 1本だけ
だと、負けが「EW の既定 α が N=8 に合っていない」で説明できてしまう。** 別の
統計を同じ器に載せてもう1本作ると、負けの原因を統計の質ではなく方式の側に
帰せる。

OWL の式は2段:

1. Wanda スコア ``A_ij = |W_ij| · ‖X_j‖₂``（W の要素の大きさ × その入力次元の
   校正上のノルム）
2. 外れ値比率 ``D_ℓ = |{A_ij > M · mean(A)}| / |A|``。M は既定 5

原典は層（transformer block）内の全 linear を対象にするが、**ここは FFN の
3枚（gate / up / down）だけで数える**。この土俵で動かしているのは FFN の配分で
あり、attention は一切触らないので、attention の重みまで混ぜると「配分の決め方」
とは別の情報が入る。原典から違えた点なので必ず書き残す。

``D_ℓ`` は M にしか依存せず、N・A・α のどれにも依存しない。GPU を使うのは
``‖X_j‖₂`` を作る前向き1回だけで、そこから先（M の格子も、スパース率の違いも）
CPU で足りる — ``ew_rule`` の CV と同じ設計である。
"""

import torch

# OWL §4 の既定。原典は {3, 5, 7, 10} を振っている
DEFAULT_M = 5.0
M_GRID = (3.0, 5.0, 7.0, 10.0)


@torch.no_grad()
def input_norms(x, batch_chunk=None, device=None):
    """``‖X_j‖₂``。``x`` は [..., 入力次元] で、最後の次元だけを残す。

    fp32 で積む。bf16 のまま二乗和を取ると、トークン数が万の桁で桁落ちする。
    """
    flat = x.reshape(-1, x.shape[-1])
    total = torch.zeros(flat.shape[-1], dtype=torch.float64)
    step = batch_chunk_rows(flat.shape[0], batch_chunk)
    for start in range(0, flat.shape[0], step):
        chunk = flat[start:start + step]
        if device is not None:
            chunk = chunk.to(device)
        total += chunk.to(torch.float32).square().sum(dim=0).double().cpu()
        del chunk
    return total.sqrt().to(torch.float32)


def batch_chunk_rows(n_rows, batch_chunk):
    """行を分ける刻み。``None`` なら分けない。"""
    return n_rows if batch_chunk is None else max(int(batch_chunk), 1)


@torch.no_grad()
def matrix_stats(weight, norms, device=None):
    """1枚の重みについて (要素数, A の総和)。2度読みの1回目。"""
    w = weight.to(device) if device is not None else weight
    scores = w.abs().to(torch.float32) * norms.to(w.device).unsqueeze(0)
    return scores.numel(), float(scores.sum().double())


@torch.no_grad()
def count_outliers(weight, norms, threshold, device=None):
    """``|{A_ij > threshold}|``。``matrix_stats`` と同じ式をもう一度回す。"""
    w = weight.to(device) if device is not None else weight
    scores = w.abs().to(torch.float32) * norms.to(w.device).unsqueeze(0)
    return int((scores > threshold).sum())


@torch.no_grad()
def layer_outlier_ratios(dense, z, h, m_grid=M_GRID, batch_chunk=None,
                         device=None):
    """1層の FFN から、M ごとの外れ値比率 ``D_ℓ``。

    ``z`` は FFN の入力（gate / up が読む）、``h`` は中間活性（down が読む）。
    どちらも [..., 次元] であればよい。

    戻り値は ``{M: D}`` と、内訳の ``detail``。**H を丸ごと持てない規模では**
    ``input_norms`` を塊ごとに自分で積んで ``ratios_from_norms`` を直に呼ぶ。
    7B の 1層で H は [32768, 11008] あり、探索と同じ校正でも 700 MiB になる。
    """
    return ratios_from_norms(
        dense,
        input_norms(z, batch_chunk=batch_chunk, device=device),
        input_norms(h, batch_chunk=batch_chunk, device=device),
        m_grid=m_grid, device=device)


@torch.no_grad()
def ratios_from_norms(dense, norm_z, norm_h, m_grid=M_GRID, device=None):
    """``‖X_j‖₂`` を先に作ってある場合の入り口。

    平均は**層の3枚をまとめて**取るので、A を保持せずに済ませるため2度読みに
    する: 先に総和と要素数だけを集め、平均が決まってから同じ式でもう一度数える。
    """
    pairs = ((dense.gate_proj.weight, norm_z),
             (dense.up_proj.weight, norm_z),
             (dense.down_proj.weight, norm_h))

    total_elements, total_sum = 0, 0.0
    for weight, norms in pairs:
        count, summed = matrix_stats(weight, norms, device=device)
        total_elements += count
        total_sum += summed
    mean = total_sum / total_elements

    ratios = {}
    for m in m_grid:
        threshold = m * mean
        outliers = sum(count_outliers(weight, norms, threshold, device=device)
                       for weight, norms in pairs)
        ratios[float(m)] = outliers / total_elements
    detail = {'mean_score': mean, 'n_elements': total_elements}
    return ratios, detail
