"""[軸6] Perplexity 評価。

CMoE-ref の ``run_cmoe.cmoe_ppl_eval`` の移送。トークン列を seqlen ごとの塊に
切り、塊ごとの平均 NLL を残す（seed 間の対応比較とブートストラップに要る）。

元実装との違いは層の置き場所だけである。元は層を1つずつ GPU へ出し入れして
いたが、ここではモデルを GPU に置いたまま進める。同じカーネルに同じ値を
入れているので数字は変わらない。
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PPLResult:
    dataset: str
    ppl: float
    mean_nll: float
    n_chunks: int
    tokens_per_chunk: int
    chunk_mean_nlls: list = field(default_factory=list)

    def as_dict(self):
        return {
            'dataset': self.dataset,
            'ppl': self.ppl,
            'mean_nll': self.mean_nll,
            'n_chunks': self.n_chunks,
            'tokens_per_chunk': self.tokens_per_chunk,
            'chunk_mean_nlls': list(self.chunk_mean_nlls),
        }


@torch.no_grad()
def evaluate_ppl(adapter, token_set, progress=None):
    """変換済み（あるいは dense の）モデルの perplexity を測る。"""
    seqlen = adapter.seqlen
    device = adapter.device
    ids = token_set.input_ids
    n_chunks = ids.numel() // seqlen
    if n_chunks < 1:
        raise ValueError(
            f'{token_set.name} は {ids.numel()} トークンで、'
            f'seqlen={seqlen} の塊が1つも取れない')

    adapter.to_device()
    ids = ids.to(device)

    hidden = torch.zeros((n_chunks, seqlen, adapter.hidden_size),
                         dtype=adapter.dtype, device=device)
    attention_mask = position_ids = None
    for index in range(n_chunks):
        chunk = ids[:, index * seqlen:(index + 1) * seqlen]
        captured = adapter.capture_layer_inputs(chunk)
        hidden[index] = captured.hidden[0]
        attention_mask = captured.attention_mask
        position_ids = captured.position_ids

    outs = torch.zeros_like(hidden)
    for layer in range(adapter.n_layers):
        for index in range(n_chunks):
            outs[index] = adapter.forward_layer(
                layer, hidden[index].unsqueeze(0), attention_mask, position_ids)
        hidden, outs = outs, hidden
        if progress is not None:
            progress(layer)

    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    chunk_mean_nlls = []
    for index in range(n_chunks):
        logits = adapter.head(hidden[index].unsqueeze(0))
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, index * seqlen:(index + 1) * seqlen][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1))
        nlls.append(loss.float() * seqlen)
        chunk_mean_nlls.append(float(loss))

    # 平均 NLL と exp は元実装と同じく float32 のまま torch で計算する。
    # python float に落としてから math.exp すると倍精度で指数を取ることになり、
    # 6桁目以降が既存の記録と食い違う。
    total = torch.stack(nlls).sum() / (n_chunks * seqlen)
    return PPLResult(
        dataset=token_set.name,
        ppl=torch.exp(total).item(),
        mean_nll=total.item(),
        n_chunks=n_chunks,
        tokens_per_chunk=seqlen,
        chunk_mean_nlls=chunk_mean_nlls,
    )
