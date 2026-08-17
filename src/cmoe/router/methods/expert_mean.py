"""方式6: expert の全メンバーの重み行を平均して router の行にする。

方式1〜4 はいずれも「どのニューロンが expert を代表するか」を答える方式だった。
この方式はメンバーから1個選ぶのをやめ、**メンバー全員の平均**を router の行に
する。推論コストは変わらない（router は依然 [n_routed, hidden] の行列2枚）。

centered 版があるのは、Llama-2-7B の gate が方向的に等方でないためである
（report/03 が共通コーンを測定）。素の平均は共通成分をそのまま残し、expert
固有の成分だけを 1/√n に薄めるので、expert 間の gate 行がほぼ平行になる。
層全体の gate 平均を引いてから平均すると、その共通成分だけが消える。
どちらも測る — どちらかを仮定はしない。

CMoE-ref の ``routerlab/methods/expert_mean.py`` の移送。
"""

import copy

import torch
import torch.nn.functional as F

from cmoe.router.methods.oracle_correlation import group_indices, validate_weights


def _pairwise_cosine(rows):
    """今作った router の行どうしの余弦（対角以外）。

    使わずに記録するだけ。平均が expert を1方向に潰したかの直接の尺度であり、
    小さな行列積1回で測れる。
    """
    if rows.shape[0] < 2:
        return ()
    unit = F.normalize(rows, p=2, dim=1)
    similarity = unit @ unit.T
    upper = torch.triu_indices(rows.shape[0], rows.shape[0], offset=1)
    return tuple(similarity[upper[0], upper[1]].tolist())


@torch.no_grad()
def expert_mean_rows(routed_groups, gate_weight, up_weight, centered=False,
                     router_normalized=True):
    """各 routed expert の gate 行・up 行を1組の router 行に平均する。

    元の dtype に関わらず float32 で足す。1376 行 × 4096 次元の float16 の和は、
    router が方向として読む桁を失う。``centered`` は層全体の gate 平均（routed
    に限らず全ニューロン）を各 gate 行から引いてから平均する。``up`` は決して
    centering しない — report/03 が方向的に等方と測っており、引くべき共通成分が
    無い。
    """
    validate_weights(gate_weight, up_weight)
    n_neurons = gate_weight.shape[0]
    device = gate_weight.device
    if not routed_groups:
        raise ValueError('expert_mean には routed expert が1個以上要る')

    gate_source = gate_weight.to(torch.float32)
    up_source = up_weight.to(torch.float32)
    global_gate_mean = gate_source.mean(dim=0)
    if centered:
        gate_source = gate_source - global_gate_mean

    gate_rows = []
    up_rows = []
    sizes = []
    for expert_index, raw_group in enumerate(routed_groups):
        group, indices = group_indices(raw_group, n_neurons, device)
        gate_rows.append(gate_source.index_select(0, indices).mean(dim=0))
        up_rows.append(up_source.index_select(0, indices).mean(dim=0))
        sizes.append(len(group))
    gate_rows = torch.stack(gate_rows)
    up_rows = torch.stack(up_rows)

    gate_norms = gate_rows.norm(dim=1)
    up_norms = up_rows.norm(dim=1)
    for name, norms in (('gate', gate_norms), ('up', up_norms)):
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
            raise ValueError(
                f'ある routed expert の平均 {name} 行が退化している: {norms.tolist()}')

    statistics = {
        'expert_sizes': sizes,
        'gate_pairwise_cosine': _pairwise_cosine(gate_rows),
        'up_pairwise_cosine': _pairwise_cosine(up_rows),
        # 1行に対して平均がどれだけ残るか。1/√n なら expert の行は方向を共有して
        # おらず平均は雑音のならしにすぎない。それより十分大きければ何か共通。
        'gate_mean_norm_ratio': float(
            gate_norms.mean() / gate_weight.to(torch.float32).norm(dim=1).mean()),
        'up_mean_norm_ratio': float(
            up_norms.mean() / up_weight.to(torch.float32).norm(dim=1).mean()),
        'gate_cosine_with_layer_mean': tuple(
            F.cosine_similarity(gate_rows, global_gate_mean[None, :]).tolist()),
    }

    if router_normalized:
        gate_rows = F.normalize(gate_rows, p=2, dim=1)
        up_rows = F.normalize(up_rows, p=2, dim=1)
    return gate_rows, up_rows, statistics


class ExpertMeanMethod:
    """方式6A: expert の行の素の平均。"""

    name = 'expert_mean'
    training_free = True
    requires_source_weights = True
    requires_profiling_stats = False
    requires_fit_z = False
    diagnostic_only = False
    centered = False

    def build(self, context, baseline):
        gate_rows, up_rows, statistics = expert_mean_rows(
            context.routed_groups,
            context.gate_weight,
            context.up_weight,
            centered=self.centered,
            router_normalized=context.router_normalized,
        )
        router = copy.deepcopy(baseline)
        router.gate.weight.data.copy_(gate_rows)
        router.classifier.weight.data.copy_(up_rows)
        # もうどのニューロンも expert を代表していない。属性を付けずに置くのでは
        # なく None を記録するのが要点で、属性が無いと共通のヘルパは分割自身の
        # 代表に落ちてしまい、このルーターが使っていない行を報告することになる。
        router.representative_indices = None
        router.selection = {
            'rule': ('expert-mean-of-centered-gate-rows-and-up-rows' if self.centered
                     else 'expert-mean-of-gate-rows-and-up-rows'),
            'centered': self.centered,
            'centering_reference': 'layer-global-gate-row-mean' if self.centered else None,
            'accumulation': 'float32',
            'source': 'dense-layer-weights-only',
            **statistics,
        }
        return router


class CenteredExpertMeanMethod(ExpertMeanMethod):
    """方式6B: 層の共通 gate コーンを除いてから同じ平均を取る。"""

    name = 'expert_mean_centered'
    centered = True
