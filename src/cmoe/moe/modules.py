"""変換後の層に載る実体: SubMLP / Router / MoE。

CMoE-ref の ``CMoE_model.py`` からの移送。演算の順序と dtype はそのままに
してある（数値が一致しないと過去の測定と比較できないため）。変更点は2つだけ:

* ``SubMLP`` は元の ``LlamaMLP`` の改名。gate/up/down と活性化関数しか持たず
  Llama 固有のものは何もないので、モデル名を冠さない。
* ``Router.extra_bias`` を buffer にした。元実装は素の属性に ``device='cuda'``
  を焼き込んでおり、``.to(device)`` で動かなかった。buffer なら
  ``persistent=False`` で state_dict を汚さずに、モジュールと一緒に移動する。
  値はどちらもゼロで、演算は変わらない。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubMLP(nn.Module):
    """FFN の一部（expert 1個分）を担う gate/up/down の三つ組。"""

    def __init__(self, hidden_size, intermediate_size, hidden_act='silu'):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == 'silu' else getattr(F, hidden_act)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Router(nn.Module):
    """代表ニューロンの行から作る、学習なしのルーター。"""

    def __init__(self, hidden_size, n_experts, n_activated, bias_speed=0.001,
                 dtype=torch.bfloat16):
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.classifier = nn.Linear(hidden_size, n_experts, bias=False)

        self.extra_scale = nn.Parameter(torch.zeros(n_experts, dtype=dtype))
        self.register_buffer('extra_bias', torch.zeros(n_experts, dtype=torch.float32),
                             persistent=False)
        self.bias_update_speed = bias_speed

    def update_bias(self, counts):
        mean_load = counts.mean()
        overloaded = counts > mean_load
        underloaded = counts < mean_load

        self.extra_bias.data[overloaded] -= self.bias_update_speed
        self.extra_bias.data[underloaded] += self.bias_update_speed

    def forward(self, x):
        scores = (self.classifier(x) * self.act_fn(self.gate(x))).abs()

        scores = scores.softmax(dim=-1, dtype=torch.float32)
        original_scores = scores
        scores = scores + self.extra_bias[None, :]

        indices = torch.topk(scores, self.topk, dim=-1)[1]

        original_scores = 1 + original_scores * self.extra_scale
        weights = original_scores.gather(1, indices)

        return weights.type_as(x), indices


class MoE(nn.Module):
    """shared expert 常時 + routed expert を Top-K で選ぶ FFN。"""

    def __init__(self, hidden_size, moe_inter_dim, n_experts, n_shared, n_activated):
        super().__init__()
        self.dim = hidden_size
        n_routed_experts = n_experts - n_shared
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = Router(hidden_size, n_routed_experts, n_activated)
        self.n_shared_experts = n_shared
        self.experts = nn.ModuleList(
            [SubMLP(self.dim, moe_inter_dim) for _ in range(n_routed_experts)])
        self.shared_experts = SubMLP(self.dim, n_shared * moe_inter_dim)

        self.cus_training = False
        self.enable_scale = True

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        if self.cus_training:
            self.gate.update_bias(counts.to(dtype=torch.bfloat16))

        counts = counts.tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if self.enable_scale:
                y[idx] += expert(x[idx]) * weights[idx, top, None]
            else:
                y[idx] += expert(x[idx])
        z = self.shared_experts(x)
        return (y + z).view(shape)


@torch.no_grad()
def forward_chunked(moe, z, residual, batch_chunk=None, device=None):
    """変換後の層の出力 ``moe(z) + residual`` を、バッチを分けて作る。

    層を丸ごと呼ばずにここで足しているのは、attention 側をすでに進めてあるから
    である（``ModelAdapter.forward_attention``）。組み立て役も配分オラクルも
    次の層への入力をこの1本で作る — 伝播が2実装あれば、探索が測った軌道と
    変換が載せた軌道が黙って分かれる。

    戻り値は z と同じデバイス。``batch_chunk`` を指定したときだけ分割して進める
    （各系列は独立に流れるので、分けても数値は変わらない）。
    """
    step = batch_chunk or z.shape[0]
    if step >= z.shape[0]:
        return moe(z) + residual
    output = torch.empty_like(z, device=z.device)
    for start in range(0, z.shape[0], step):
        stop = min(start + step, z.shape[0])
        value = moe(z[start:stop].to(device)) + residual[start:stop].to(device)
        output[start:stop].copy_(value.to(z.device))
    return output
