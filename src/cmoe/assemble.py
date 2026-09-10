"""組み立て役: Adapter × Data × Carve × Alloc × Router → 変換済みモデル。

軸どうしを繋ぐ知識はここにしか無い。分割方式はルーター方式を知らず、ルーター
方式は配分を知らない。両方を同時に入れたときに何が起きるかを知っているのは
この1本だけである。

層の処理順は CMoE-ref の ``cmoe_sequential`` / ``construct_moe`` と同じ:
層0 の入力を捕まえ、層ごとに attention を進め、その層の FFN 入力で活性を
プロファイルし、分割してルーターを作り、MoE に差し替え、その出力を次の層の
入力にする。前の層が変換済みの状態で次の層を測る、という順序が数値に効くので
変えていない。

ルーター方式は**複数まとめて**作れる。方式4 が先行方式の代表集合を出発点に
するため必要であると同時に、同じ carve の上でルーターだけを差し替えて比べる
（report/10・13 のやり方）ためでもある。

このとき層に載せて次層へ伝播させるのは、常に**先頭の方式（現行 CMoE）**の
ルーターである。どの方式も同じ軌道の上で作られ・評価されることになり、これが
既存の測定の条件でもある。他の方式のルーターは ``install_routers`` で後から
差し替える。
"""

from dataclasses import dataclass, field
import time

import torch
import torch.nn as nn

from cmoe.alloc.oracles.base import CarvedLayer
from cmoe.carve.base import build_experts
from cmoe.carve.profile import profile_layer, select_positions
from cmoe.moe.modules import MoE, forward_chunked
from cmoe.router.base import (build_baseline_router, build_router, make_context,
                              router_representative_indices)
from cmoe.router.diagnostics import evaluate_routers_against_abs_oracle


@dataclass
class LayerRecord:
    """1層分の変換記録。数字が合わないときの二分探索に使う。"""

    layer: int
    n_shared: int
    topk: int
    expert_sizes: tuple
    baseline_representatives: tuple
    representatives: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self):
        return {
            'layer': self.layer,
            'n_shared': self.n_shared,
            'topk': self.topk,
            'expert_sizes': list(self.expert_sizes),
            'baseline_representatives': list(self.baseline_representatives),
            'representatives': {
                name: (list(row) if row is not None else None)
                for name, row in self.representatives.items()
            },
            'selections': self.selections,
            'diagnostics': self.diagnostics,
            'seconds': self.seconds,
        }


@dataclass
class ConversionReport:
    allocation: object
    carver: str
    router_methods: tuple
    n_experts: int
    routers: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def baseline_method(self):
        return self.router_methods[0]

    def as_dict(self):
        return {
            'allocation': self.allocation.metadata(),
            'carver': self.carver,
            'router_methods': list(self.router_methods),
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


def layer_factory(carver, n_experts, bias_speed=0.001, router_norm=True,
                  device=None, router_method=None):
    """配分オラクルへ渡す「x → その層に載る MoE」。

    配分の探索は、候補 x の層が実際にどう振る舞うかを見なければ採点できないが、
    分割規則もルーター方式もその関心事ではない。ここで包んで渡すことで、
    ``alloc`` が ``router`` を import せずに済む — 軸4 と軸5 を同時に使う知識は、
    組み立て役だけが持つ。

    層に載せるのと同じ経路（``build_baseline_router`` → ``build_moe``）を通る。
    探索が測った層と、あとで ``Converter`` が載せる層が別物にならないための
    取り決めである。

    ``router_method`` を渡すと、候補の層にその方式のルーターが載る。**配分と
    ルーターを揃えて探索する**ときに使う。既定の None では基準ルーター（現行
    CMoE）が載り、これが report/07・08 の探索が通った経路である。配分が方式1
    の前提で選ばれていることは、あとで別の方式を載せたときの食い違いになる。

    方式が読む ``fit_z`` には**探索が持っている z（校正データの、その層での
    活性）**を渡す。``Converter`` が使う独立した fit セットは探索の中に無い。
    Top-K=0 の層では方式を呼ばない — routed のループが一度も回らないので、
    どの方式も出力を1ビットも変えられない（``Converter._build_routers`` と
    同じ断り方である）。
    """
    @torch.no_grad()
    def build(dense, rates, markers, n_shared, topk, z=None, layer=None):
        partition = carver.carve(dense, rates, markers, n_shared, z=z,
                                 layer=layer)
        router = build_baseline_router(dense, partition, topk,
                                       bias_speed=bias_speed,
                                       normalize=router_norm)
        if router_method is not None and topk > 0:
            context = make_context(
                None, dense, partition, topk, router_method,
                rates=rates, markers=markers, fit_z=z, normalize=router_norm)
            router = build_router(router_method, context, router)
        moe = build_moe(dense, partition, topk, router, n_experts)
        # expert とルーターの行は dense の重みから来るので既に層と同じデバイスに
        # ある。ゼロで作った extra_scale / extra_bias だけが取り残される
        if device is not None:
            moe.to(device)
        return CarvedLayer(n_shared=n_shared, topk=topk, moe=moe,
                           partition=partition)

    return build


def install_routers(adapter, routers):
    """変換済みモデルのルーターだけを差し替える。expert には触れない。

    これが「ルーターだけを比べる」実験の要である。分割も expert 重みも Top-K も
    そのままで、``MoE.gate`` だけが入れ替わる。
    """
    # 部分変換（--layers）のときは変換した層の分しか無いので、多すぎる場合だけ拒む
    if len(routers) > adapter.n_layers:
        raise ValueError(
            f'ルーターが {len(routers)} 個、モデルは {adapter.n_layers} 層')
    for index, router in enumerate(routers):
        moe = adapter.layers[index].mlp
        if not isinstance(moe, MoE):
            raise ValueError(f'層 {index} は変換されていない')
        if router.dim != moe.gate.dim or router.topk != moe.gate.topk:
            raise ValueError(f'層 {index}: 差し替えでルーターの次元が変わる')
        moe.gate = router
    return adapter


class Converter:
    """1つのモデルを、指定された配分とルーター方式で変換する。

    変換は破壊的である（層の FFN を置き換える）。同じモデルを2通りに変換する
    ことはできないので、構成ごとにモデルを読み直す。
    """

    def __init__(self, adapter, carver, router_methods, n_experts,
                 k_act=10, bias_speed=0.001, profiling_norm=True,
                 router_norm=True, batch_chunk=None, fit_batch_chunk=4,
                 token_chunk=4096, n_layers=None, scored_positions_only=False,
                 log=None):
        self.adapter = adapter
        self.carver = carver
        # 先頭が層に載る方式であり、次層への伝播もこれで行う
        self.router_methods = list(router_methods)
        if not self.router_methods:
            raise ValueError('ルーター方式が1つも指定されていない')
        self.n_experts = n_experts
        self.k_act = k_act
        self.bias_speed = bias_speed
        self.profiling_norm = profiling_norm
        self.router_norm = router_norm
        self.batch_chunk = batch_chunk
        # fit / validation は 64 系列あるので、常に分けて進める
        self.fit_batch_chunk = fit_batch_chunk
        self.token_chunk = token_chunk
        # 先頭 n_layers 層だけを変換する。動作確認用で、残りは dense のまま
        self.n_layers = n_layers
        # 活性を数える位置を、採点に効く位置だけに絞る。校正セットが印
        # （``TokenSet.segments``）を持っているときにしか立てられない
        self.scored_positions_only = scored_positions_only
        self.log = log or (lambda message='': None)

    @property
    def needs_fit(self):
        return any(getattr(method, 'requires_fit_z', False)
                   for method in self.router_methods)

    @torch.no_grad()
    def convert(self, carve, allocation, fit=None, validation=None):
        adapter = self.adapter
        n_layers = self.n_layers or adapter.n_layers
        if n_layers > adapter.n_layers:
            raise ValueError(
                f'{n_layers} 層を変換しようとしているが、モデルは '
                f'{adapter.n_layers} 層')
        allocation.check_layers(n_layers)
        if self.needs_fit and fit is None:
            raise ValueError(
                'このルーター方式は fit データを要求する（--calib の分割を渡す）')

        profile_mask = self._profile_mask(carve)

        started = time.time()
        carve_inputs = adapter.capture_layer_inputs(carve.input_ids)
        fit_inputs = (adapter.capture_layer_inputs(fit.input_ids)
                      if fit is not None else None)
        validation_inputs = (adapter.capture_layer_inputs(validation.input_ids)
                             if validation is not None else None)
        adapter.to_device()

        hidden = carve_inputs.hidden
        if self.batch_chunk is None:
            hidden = hidden.to(adapter.device)
        fit_hidden = fit_inputs.hidden if fit_inputs is not None else None
        validation_hidden = (validation_inputs.hidden
                             if validation_inputs is not None else None)

        names = tuple(method.name for method in self.router_methods)
        report = ConversionReport(
            allocation=allocation, carver=self.carver.name,
            router_methods=names, n_experts=self.n_experts,
            routers={name: [] for name in names})

        for index in range(n_layers):
            layer_started = time.time()
            n_shared = allocation[index]
            topk = allocation.topk(index)

            dense = adapter.dense_ffn(index)
            z, residual = adapter.forward_attention(
                index, hidden, carve_inputs.attention_mask,
                carve_inputs.position_ids, batch_chunk=self.batch_chunk)

            # 分割が読むのはここだけ。次層への伝播は絞っていない z で行う
            profile_z = select_positions(z, profile_mask)
            rates, markers = self._profile(dense, profile_z)
            partition = self.carver.carve(dense, rates, markers, n_shared,
                                          z=profile_z, layer=index)
            baseline = build_baseline_router(
                dense, partition, topk, bias_speed=self.bias_speed,
                normalize=self.router_norm)

            fit_z = fit_residual = None
            validation_z = validation_residual = None
            if fit_hidden is not None:
                fit_z, fit_residual = adapter.forward_attention(
                    index, fit_hidden, fit_inputs.attention_mask,
                    fit_inputs.position_ids, batch_chunk=self.fit_batch_chunk)
            if validation_hidden is not None:
                validation_z, validation_residual = adapter.forward_attention(
                    index, validation_hidden, validation_inputs.attention_mask,
                    validation_inputs.position_ids,
                    batch_chunk=self.fit_batch_chunk)

            routers, record = self._build_routers(
                index, dense, partition, topk, baseline, rates, markers,
                fit_z, validation_z)
            record.expert_sizes = tuple(
                len(group) for group in partition.expert_groups)
            for name in names:
                report.routers[name].append(routers[name])

            moe = build_moe(dense, partition, topk, routers[names[0]],
                            self.n_experts)
            # expert とルーターの行は dense の重みから来るので、その時点で層と
            # 同じデバイスに載っている。載っていないのはゼロで作った
            # extra_scale / extra_bias だけで、それをここで揃える。
            moe.to(adapter.device)
            adapter.replace_ffn(index, moe)

            hidden = self._forward_moe(moe, z, residual, self.batch_chunk)
            if fit_hidden is not None:
                fit_hidden = self._forward_moe(
                    moe, fit_z, fit_residual, self.fit_batch_chunk)
            if validation_hidden is not None:
                validation_hidden = self._forward_moe(
                    moe, validation_z, validation_residual, self.fit_batch_chunk)

            record.seconds = time.time() - layer_started
            report.layers.append(record)
            self.log(self._layer_line(record))

            del (dense, z, profile_z, residual, rates, markers, partition,
                 baseline, fit_z, fit_residual, validation_z,
                 validation_residual, routers)
            torch.cuda.empty_cache()

        report.seconds = time.time() - started
        return report

    def _profile_mask(self, carve):
        """活性を数える位置。絞らないときは None。

        絞れと言われたのに校正セットが印を持たないのは、素の文章のセットを
        1問1系列の設定で走らせているということなので、黙って全位置に落とさず
        断る。
        """
        if not self.scored_positions_only:
            return None
        mask = carve.scored_mask()
        if mask is None:
            raise ValueError(
                f'{carve.name} は採点位置の印を持たない。'
                '印を持つ校正セット（benchqa）を使うこと')
        return mask

    def _layer_line(self, record):
        line = (f'層 {record.layer:>2}: x={record.n_shared} Top-K={record.topk}')
        for name, values in record.diagnostics.items():
            line += f' {name} R={values["router_r"]:.6f}'
        return line + f' ({record.seconds:.1f}s)'

    @torch.no_grad()
    def _build_routers(self, index, dense, partition, topk, baseline,
                       rates, markers, fit_z, validation_z):
        """この層のルーターを、方式ごとに1つずつ作る。

        Top-K=0 の層ではどの方式も出力を1ビットも変えられない（routed の
        ループが一度も回らない）ので、基準ルーターだけを作って全方式に同じ
        モジュールを渡す。探索も診断も走らせない — 何も選ばない以上、回収率は
        全方式で shared 質量に等しく、オラクルとの差は恒等的に 0 である。
        """
        names = tuple(method.name for method in self.router_methods)
        record = LayerRecord(
            layer=index, n_shared=partition.n_shared, topk=topk,
            expert_sizes=(),
            baseline_representatives=partition.representative_indices)

        if topk == 0:
            routers = {name: baseline for name in names}
            record.representatives = {
                name: partition.representative_indices for name in names}
            return routers, record

        routers = {}
        initial_sets = []
        rows = {}
        for method in self.router_methods:
            context = make_context(
                index, dense, partition, topk, method,
                rates=rates, markers=markers, fit_z=fit_z,
                normalize=self.router_norm,
                initial_sets=self._starting_sets(method, initial_sets, rows))
            router = build_router(method, context, baseline)
            routers[method.name] = router

            row = router_representative_indices(router, partition)
            record.representatives[method.name] = row
            rows[method.name] = row
            selection = getattr(router, 'selection', None)
            if selection is not None:
                record.selections[method.name] = selection
            # 出発点を集めるのは、初期集合を要求しない・配備できる方式だけ
            if (row is not None
                    and not getattr(method, 'requires_initial_sets', False)
                    and not getattr(method, 'diagnostic_only', False)
                    and row not in initial_sets):
                initial_sets.append(row)

        if validation_z is not None:
            # score を名乗れるルーターだけを診断にかける。代表ニューロン型は
            # gate / classifier の積がその score で、それ以外の族は
            # ``routing_scores`` で名乗る。診断用オラクルはどちらも持たない
            # ので自然に外れる
            diagnostic = {
                name: router for name, router in routers.items()
                if hasattr(router, 'routing_scores')
                or (hasattr(router, 'gate') and hasattr(router, 'classifier'))}
            if diagnostic:
                record.diagnostics = evaluate_routers_against_abs_oracle(
                    diagnostic, validation_z, partition.expert_groups,
                    dense.gate_proj.weight.detach(),
                    dense.up_proj.weight.detach(),
                    chunk_size=self.token_chunk)
        return routers, record

    def _starting_sets(self, method, initial_sets, rows):
        """その方式に渡す初期代表集合。

        既定は「先行方式が選んだ相異なる集合すべて」で、方式4 はそのそれぞれから
        探索する。``frozen_source`` を宣言した方式（方式5）は代表を1つも動かさず、
        名指しした方式の代表を**そのまま凍結する**ので、渡すのは1本だけである。
        どの方式を凍結したかが方式の側の宣言になり、組み立て役に方式ごとの分岐が
        増えない。
        """
        source = getattr(method, 'frozen_source', None)
        if source is None:
            return tuple(initial_sets)
        if source not in rows:
            raise ValueError(
                f'{method.name} は {source!r} の代表を凍結するが、'
                f'{source!r} がこの構築順の前に居ない')
        if rows[source] is None:
            raise ValueError(
                f'{method.name} は {source!r} の代表を凍結するが、'
                f'{source!r} は代表を持たない')
        return (rows[source],)

    @torch.no_grad()
    def _profile(self, dense, z):
        """FFN 入力から活性統計を作る。バッチを分けても結果は同じ。"""
        return profile_layer(dense, z, k_act=self.k_act,
                             normalize=self.profiling_norm,
                             batch_chunk=self.batch_chunk,
                             device=self.adapter.device)

    @torch.no_grad()
    def _forward_moe(self, moe, z, residual, batch_chunk):
        return forward_chunked(moe, z, residual, batch_chunk=batch_chunk,
                               device=self.adapter.device)
