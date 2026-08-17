"""[軸4] ルーター方式の境界。

ルーター方式が観測してよいものと、返してよいものを固定する。分割・x・Top-K は
すでに決まっており、方式はそれらに触れない。``build_router`` がその不変を
検査してから層に渡す。

CMoE-ref の ``routerlab/contracts.py`` と ``routerlab/candidate.py`` の移送。
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmoe.moe.modules import Router


@dataclass(frozen=True)
class RouterBuildContext:
    """ルーターが見てよい、固定済みの CMoE 構造。"""

    layer_index: int
    hidden_size: int
    n_experts: int
    n_shared: int
    topk: int
    expert_groups: tuple
    representative_indices: tuple
    gate_weight: torch.Tensor | None = None
    up_weight: torch.Tensor | None = None
    activation_rates: torch.Tensor | None = None
    activation_markers: torch.Tensor | None = None
    fit_z: torch.Tensor | None = None
    router_normalized: bool = True
    # 先行方式がすでに選んだ代表集合。共同探索が出発点として使う。
    initial_representative_sets: tuple = ()
    # expert ごとの候補制限。空なら expert 内の全ニューロンが候補。
    candidate_pools: tuple = ()

    @property
    def routed_groups(self):
        return self.expert_groups[1:]

    @property
    def n_routed(self):
        return self.n_experts - self.n_shared


class RouterMethod(Protocol):
    """固定済みの候補に対して、配備できるルーターを1つ作る。"""

    name: str
    training_free: bool

    def build(self, context: RouterBuildContext, baseline: nn.Module) -> nn.Module:
        ...


@torch.no_grad()
def build_baseline_router(dense, partition, topk, bias_speed=0.001,
                          normalize=True):
    """代表ニューロンの行から現行 CMoE のルーターを作る。

    ``Router.classifier`` に up の行、``Router.gate`` に gate の行が入る。
    既定では両方を L2 正規化する（CMoE-ref の ``--no-router-norm`` 無しの枝）。
    """
    n_routed = partition.n_routed
    if n_routed < 1:
        raise ValueError(
            f'n_shared={partition.n_shared} では routed expert が残らない')
    if not 0 <= topk <= n_routed:
        raise ValueError(f'Top-K={topk} は routed expert {n_routed} 個に対して範囲外')

    router = Router(dense.hidden_size, n_routed, topk, bias_speed=bias_speed)
    core = list(partition.representative_indices)
    core_up = dense.up_proj.weight.data[core, :]
    core_gate = dense.gate_proj.weight.data[core, :]
    if normalize:
        router.classifier.weight.data = F.normalize(core_up, p=2, dim=1)
        router.gate.weight.data = F.normalize(core_gate, p=2, dim=1)
    else:
        router.classifier.weight.data = core_up
        router.gate.weight.data = core_gate
    return router


_UNSET = object()


def router_representative_indices(router, partition):
    """そのルーターが実際に使っているニューロン行。

    ``None`` は「どのニューロンの行でもない」（集約方式が作った）ことを指し、
    属性に触れなかった方式（分割の代表をそのまま使う）とは区別する。
    """
    value = getattr(router, 'representative_indices', _UNSET)
    if value is _UNSET:
        return tuple(partition.representative_indices)
    if value is None:
        return None
    return tuple(value)


@torch.no_grad()
def build_router(method, context, baseline):
    """方式にルーターを作らせ、構造を変えていないことを検査して返す。"""
    router = method.build(context, baseline)
    if not isinstance(router, nn.Module):
        raise TypeError(
            f'ルーター方式が {type(router).__name__} を返した (nn.Module のはず)')
    if getattr(router, 'dim', None) != context.hidden_size:
        raise ValueError(
            f'ルーターの dim {getattr(router, "dim", None)} != {context.hidden_size}')
    if getattr(router, 'topk', None) != context.topk:
        raise ValueError(
            f'ルーターの topk {getattr(router, "topk", None)} != {context.topk}')
    return router


def make_context(layer_index, dense, partition, topk, method,
                 rates=None, markers=None, fit_z=None, normalize=True,
                 initial_sets=(), candidate_pools=()):
    """方式が要求したものだけを載せた context を組む。

    要求していない重みや統計は渡さない。実験方式が dense の重みへの書き込み
    可能な参照を受け取ることは無い（detach + clone してから渡す）。
    """
    needs_weights = bool(getattr(method, 'requires_source_weights', False))
    needs_stats = bool(getattr(method, 'requires_profiling_stats', False))
    needs_fit_z = bool(getattr(method, 'requires_fit_z', False))
    needs_initial = bool(getattr(method, 'requires_initial_sets', False))

    if needs_stats and (rates is None or markers is None):
        raise ValueError(f'ルーター方式 {method.name!r} は活性統計を要求している')
    if needs_fit_z and fit_z is None:
        raise ValueError(f'ルーター方式 {method.name!r} は fit_z を要求している')
    if needs_initial and not initial_sets:
        raise ValueError(f'ルーター方式 {method.name!r} は初期代表集合を要求している')

    return RouterBuildContext(
        layer_index=layer_index,
        hidden_size=dense.hidden_size,
        n_experts=partition.n_experts,
        n_shared=partition.n_shared,
        topk=topk,
        expert_groups=partition.expert_groups,
        representative_indices=partition.representative_indices,
        gate_weight=(dense.gate_proj.weight.detach().clone() if needs_weights else None),
        up_weight=(dense.up_proj.weight.detach().clone() if needs_weights else None),
        activation_rates=(rates.detach() if needs_stats else None),
        activation_markers=(markers.detach() if needs_stats else None),
        fit_z=(fit_z.detach() if needs_fit_z else None),
        router_normalized=normalize,
        initial_representative_sets=(
            tuple(tuple(int(i) for i in row) for row in initial_sets)
            if needs_initial else ()),
        candidate_pools=(
            tuple(tuple(int(i) for i in row) for row in candidate_pools)
            if needs_initial else ()),
    )
