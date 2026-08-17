"""組み立て役: Adapter × Data × Carve × Alloc × Router → 変換済みモデル。

軸どうしを繋ぐ知識はここにしか無い。分割方式はルーター方式を知らず、ルーター
方式は配分を知らない。両方を同時に入れたときに何が起きるかを知っているのは
この1本だけである。

層の処理順は CMoE-ref の ``cmoe_sequential`` / ``construct_moe`` と同じ:
層0 の入力を捕まえ、層ごとに attention を進め、その層の FFN 入力で活性を
プロファイルし、分割してルーターを作り、MoE に差し替え、その出力を次の層の
入力にする。前の層が変換済みの状態で次の層を測る、という順序が数値に効くので
変えていない。
"""

from dataclasses import dataclass, field
import time

import torch
import torch.nn as nn

from cmoe.carve.base import build_experts
from cmoe.carve.profile import analyze_activations, hidden_activations
from cmoe.moe.modules import MoE
from cmoe.router.base import (build_baseline_router, build_router, make_context,
                              router_representative_indices)


@dataclass
class LayerRecord:
    """1層分の変換記録。数字が合わないときの二分探索に使う。"""

    layer: int
    n_shared: int
    topk: int
    expert_sizes: tuple
    representatives: tuple
    baseline_representatives: tuple
    seconds: float

    def as_dict(self):
        return {
            'layer': self.layer,
            'n_shared': self.n_shared,
            'topk': self.topk,
            'expert_sizes': list(self.expert_sizes),
            'representatives': (list(self.representatives)
                                if self.representatives is not None else None),
            'baseline_representatives': list(self.baseline_representatives),
            'seconds': self.seconds,
        }


@dataclass
class ConversionReport:
    allocation: object
    carver: str
    router_method: str
    n_experts: int
    layers: list = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self):
        return {
            'allocation': self.allocation.metadata(),
            'carver': self.carver,
            'router_method': self.router_method,
            'n_experts': self.n_experts,
            'seconds': self.seconds,
            'layers': [row.as_dict() for row in self.layers],
        }


@torch.no_grad()
def build_moe(dense, partition, topk, router, n_experts):
    """分割・ルーター・重みを1つの MoE モジュールに組む。"""
    moe = MoE(dense.hidden_size, dense.intermediate_size // n_experts,
              n_experts, partition.n_shared, topk)
    experts = build_experts(dense, partition)
    moe.gate = router
    moe.experts = nn.ModuleList(experts[1:])
    moe.shared_experts = experts[0]
    moe.cus_training = False
    return moe


class Converter:
    """1つのモデルを、指定された配分とルーター方式で変換する。

    変換は破壊的である（層の FFN を置き換える）。同じモデルを2通りに変換する
    ことはできないので、構成ごとにモデルを読み直す。
    """

    def __init__(self, adapter, carver, router_method, n_experts,
                 k_act=10, bias_speed=0.001, profiling_norm=True,
                 router_norm=True, batch_chunk=None, n_layers=None, log=None):
        self.adapter = adapter
        # 先頭 n_layers 層だけを変換する。動作確認用で、残りは dense のまま
        self.n_layers = n_layers
        self.carver = carver
        self.router_method = router_method
        self.n_experts = n_experts
        self.k_act = k_act
        self.bias_speed = bias_speed
        self.profiling_norm = profiling_norm
        self.router_norm = router_norm
        self.batch_chunk = batch_chunk
        self.log = log or (lambda message='': None)

    @torch.no_grad()
    def convert(self, calibration, allocation):
        adapter = self.adapter
        n_layers = self.n_layers or adapter.n_layers
        if n_layers > adapter.n_layers:
            raise ValueError(
                f'{n_layers} 層を変換しようとしているが、モデルは '
                f'{adapter.n_layers} 層')
        allocation.check_layers(n_layers)

        started = time.time()
        inputs = adapter.capture_layer_inputs(calibration.input_ids)
        adapter.to_device()

        hidden = inputs.hidden
        if self.batch_chunk is None:
            hidden = hidden.to(adapter.device)

        report = ConversionReport(
            allocation=allocation, carver=self.carver.name,
            router_method=self.router_method.name, n_experts=self.n_experts)

        for index in range(n_layers):
            layer_started = time.time()
            n_shared = allocation[index]
            topk = allocation.topk(index)

            dense = adapter.dense_ffn(index)
            z, residual = adapter.forward_attention(
                index, hidden, inputs.attention_mask, inputs.position_ids,
                batch_chunk=self.batch_chunk)

            rates, markers = self._profile(dense, z)
            partition = self.carver.carve(dense, rates, markers, n_shared)

            baseline = build_baseline_router(
                dense, partition, topk, bias_speed=self.bias_speed,
                normalize=self.router_norm)
            context = make_context(
                index, dense, partition, topk, self.router_method,
                rates=rates, markers=markers, normalize=self.router_norm)
            router = build_router(self.router_method, context, baseline)

            moe = build_moe(dense, partition, topk, router, self.n_experts)
            # expert とルーターの行は dense の重みから来るので、その時点で層と
            # 同じデバイスに載っている。載っていないのはゼロで作った
            # extra_scale / extra_bias だけで、それをここで揃える。元実装は
            # 生成時に 'cuda' を焼き込んでいた箇所にあたる。
            moe.to(adapter.device)
            adapter.replace_ffn(index, moe)

            hidden = self._forward_moe(moe, z, residual)

            report.layers.append(LayerRecord(
                layer=index,
                n_shared=n_shared,
                topk=topk,
                expert_sizes=tuple(len(group) for group in partition.expert_groups),
                representatives=router_representative_indices(router, partition),
                baseline_representatives=partition.representative_indices,
                seconds=time.time() - layer_started,
            ))
            self.log(f'層 {index:>2}: x={n_shared} Top-K={topk} '
                     f'({report.layers[-1].seconds:.1f}s)')

            del dense, z, residual, rates, markers, partition, baseline, context
            torch.cuda.empty_cache()

        report.seconds = time.time() - started
        return report

    @torch.no_grad()
    def _profile(self, dense, z):
        """FFN 入力から活性統計を作る。バッチを分けても結果は同じ。"""
        step = self.batch_chunk or z.shape[0]
        rows = []
        for start in range(0, z.shape[0], step):
            chunk = z[start:start + step].to(self.adapter.device)
            h = hidden_activations(dense, chunk, normalize=self.profiling_norm)
            rows.append(h.to('cpu'))
            del h, chunk
        h = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        _, rates, markers = analyze_activations(h, k_act=self.k_act)
        return rates, markers

    @torch.no_grad()
    def _forward_moe(self, moe, z, residual):
        step = self.batch_chunk or z.shape[0]
        if step >= z.shape[0]:
            return moe(z) + residual
        output = torch.empty_like(z, device=z.device)
        for start in range(0, z.shape[0], step):
            stop = min(start + step, z.shape[0])
            value = (moe(z[start:stop].to(self.adapter.device))
                     + residual[start:stop].to(self.adapter.device))
            output[start:stop].copy_(value.to(z.device))
        return output
