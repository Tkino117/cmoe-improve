"""診断用: 真の |h| 活性量を読んで expert を選ぶオラクル。

配備できるルーターと違い、選ぶ前に routed 側の全ニューロンの活性を計算する。
**回収率のオラクルではあるが、PPL のオラクルではない** — down projection と
後続層への伝播があるので、質量最大が PPL 最小とは限らない。

推論の高速化には使えない（選択の前に、その層がこれから計算する H を先に
計算してしまう）。現行ルーターとの差が、選択を理想化したときに取り戻せる量の
上限になる。

CMoE-ref の ``routerlab/methods/oracle_abs.py`` の移送。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OracleAbsRouter(nn.Module):
    """トークンごとの L1 活性量が最大の routed expert を選ぶ。"""

    def __init__(self, gate_weight, up_weight, topk):
        super().__init__()
        if gate_weight.dim() != 3 or up_weight.dim() != 3:
            raise ValueError('オラクルの重みは [experts, neurons, hidden] のはず')
        if gate_weight.shape != up_weight.shape:
            raise ValueError(
                f'gate/up の形が違う: {tuple(gate_weight.shape)} と '
                f'{tuple(up_weight.shape)}')
        n_routed, neurons_per_expert, hidden_size = gate_weight.shape
        if n_routed < 1 or neurons_per_expert < 1 or hidden_size < 1:
            raise ValueError(f'オラクルの重みが空: {gate_weight.shape}')
        if not 0 <= topk <= n_routed:
            raise ValueError(f'topk={topk} は routed expert {n_routed} 個に対して範囲外')

        self.dim = hidden_size
        self.topk = topk
        self.n_routed = n_routed
        # すでに切り出された expert 行の固定コピー。buffer にしておくと、
        # 診断用オラクルを学習対象にせずにデバイス/dtype の移動が層に追従する。
        self.register_buffer('gate_weight', gate_weight.detach().clone())
        self.register_buffer('up_weight', up_weight.detach().clone())

    @classmethod
    def from_experts(cls, experts, topk):
        experts = list(experts)
        if not experts:
            raise ValueError('オラクルには routed expert が1個以上要る')
        gate = torch.stack([expert.gate_proj.weight.detach() for expert in experts])
        up = torch.stack([expert.up_proj.weight.detach() for expert in experts])
        return cls(gate, up, topk)

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError(f'x の最終次元 {x.shape[-1]}、オラクルは {self.dim} を期待')
        x = x.reshape(-1, self.dim)
        if self.topk == 0:
            indices = torch.empty((x.shape[0], 0), dtype=torch.long, device=x.device)
            return x.new_ones(indices.shape), indices

        # expert ごとに進めるので、[tokens, 全 routed ニューロン] ではなく
        # 1グループ分の H だけを実体化する。
        scores = torch.empty(
            (x.shape[0], self.n_routed), dtype=torch.float32, device=x.device)
        for index in range(self.n_routed):
            h = F.silu(F.linear(x, self.gate_weight[index]))
            h.mul_(F.linear(x, self.up_weight[index]))
            scores[:, index] = h.abs_().sum(dim=1, dtype=torch.float32)
        indices = scores.topk(self.topk, dim=1).indices
        return x.new_ones(indices.shape), indices


class OracleAbsExpertRouter(nn.Module):
    """組み上がった MoE 用の、メモリを増やさないオラクル。

    expert モジュールがすでに重みを持っているので、数 GB のコピーをもう1つ
    登録せず参照だけを持つ。デバイス/dtype の移動は外側の MoE の責任。
    """

    def __init__(self, experts, topk):
        super().__init__()
        experts = tuple(experts)
        if not experts:
            raise ValueError('オラクルには routed expert が1個以上要る')
        hidden_sizes = {expert.gate_proj.in_features for expert in experts}
        intermediate_sizes = {expert.gate_proj.out_features for expert in experts}
        if len(hidden_sizes) != 1 or len(intermediate_sizes) != 1:
            raise ValueError('routed expert の入力次元と中間次元は共通のはず')
        if not 0 <= topk <= len(experts):
            raise ValueError(f'topk={topk} は expert {len(experts)} 個に対して範囲外')
        self.dim = hidden_sizes.pop()
        self.topk = topk
        # expert を2つ目のモジュール経路の下に登録しない。参照は保たれ、
        # MoE が実際に実行するのと同じオブジェクトのままである。
        object.__setattr__(self, '_experts', experts)

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError(f'x の最終次元 {x.shape[-1]}、オラクルは {self.dim} を期待')
        x = x.reshape(-1, self.dim)
        if self.topk == 0:
            indices = torch.empty((x.shape[0], 0), dtype=torch.long, device=x.device)
            return x.new_ones(indices.shape), indices

        scores = torch.empty(
            (x.shape[0], len(self._experts)), dtype=torch.float32, device=x.device)
        for index, expert in enumerate(self._experts):
            h = expert.act_fn(expert.gate_proj(x))
            h.mul_(expert.up_proj(x))
            scores[:, index] = h.abs_().sum(dim=1, dtype=torch.float32)
        indices = scores.topk(self.topk, dim=1).indices
        return x.new_ones(indices.shape), indices


class OracleAbsMethod:
    name = 'oracle_abs'
    training_free = True
    requires_source_weights = True
    diagnostic_only = True

    def build(self, context, baseline):
        if context.gate_weight is None or context.up_weight is None:
            raise ValueError('oracle_abs は dense の gate/up 重みを要求する')
        routed = context.routed_groups
        sizes = {len(group) for group in routed}
        if len(sizes) != 1 or not sizes or 0 in sizes:
            raise ValueError('routed expert のグループは共通の非零サイズのはず')
        gate = torch.stack([context.gate_weight[list(group)] for group in routed])
        up = torch.stack([context.up_weight[list(group)] for group in routed])
        return OracleAbsRouter(gate, up, context.topk)
