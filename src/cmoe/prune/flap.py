"""[対照] FLAP (An et al., AAAI 2024) の移送。**提案手法ではない。**

ExpertWeaver Table 2 の training-free 比較手法のひとつ。公式実装
(github.com/CASIA-IVA-Lab/FLAP) の ``lib/prune.py: prune_flap`` と
``lib/layerwrapper.py: BiasGPT`` を、この基盤のアダプタとデータ軸の上に移した。

公式実装をそのまま持ち込めないのは、あちらが自前の ``modeling_llama.py`` を同梱して
古い transformers に固定しているためである（このリポジトリは 4.47.1 に固定）。
移したのは統計と規則だけで、EW-rule・OWL・LExI を式だけ移したのと同じやり方である。

規則は4段:

1. **fluctuation 統計** … 各 Linear の入力チャネルごとに、平均 ``baseline_inp`` と
   ゆらぎ ``fluc_inp`` を校正データ上で貯める。統計は**刈る前のモデル**の上で
   層ごとに取り、出力をそのまま次の層の入力にする（原典と同じ順序）
2. **WIFV** … ``fluc_inp × Σ_out W²``。attention 側はこれを2乗する
3. **層をまたいだ標準化と大域しきい値 (AL-AM)** … 層ごとに標準化してから全層・
   全構造を1本に並べ、パラメータ数で重み付けした累積が目標に達する所で切る。
   **層ごとに残す本数が変わる**のがこの手法の肝で、``scope='mlp'`` ではこれが
   そのまま「層別に FFN の刈り数を配る規則」になり、report/22 の LExI・OWL と
   同じ位置の対照になる
4. **bias 補償** … 落とした入力チャネルの平均寄与を出力側の bias に移す

**原典との違いは1点だけ。** 公式実装の ``compression_weight`` は
``torch.ones_like(indices)`` すなわち int64 テンソルに ``512.0/3`` を代入しており、
170.667 が 170 に切り捨てられる。ヘッド1本と FFN ニューロン1本のパラメータ比は
``4·d·128 / 3·d = 512/3`` なので、ここは浮動小数のまま使う。切り捨てのままだと
目標スパース率が 0.4% ずれる。
"""

import torch

from cmoe.prune.base import (LayerPlan, PrunePlan, capture_inputs,
                             check_prunable)

# 原典の既定。--metrics WIFV --structure AL-AM
DEFAULT_SAMPLES = 2048
DEFAULT_SEQLEN = 128


class _BiasStats:
    """``BiasGPT`` の移送。1系列ずつ渡す前提で、実効の batch_size は常に1。

    貯め方（分母の取り方）は原典そのままにしてある。統計として素直な形では
    ないが、ここで直すと移送ではなくなる。累積は fp32 で持つ — bf16 の
    仮数8ビットでは、2048本ぶんの平均を貯める側が先に潰れる。
    """

    def __init__(self, in_dim, device):
        self.baseline_inp = torch.zeros(in_dim, dtype=torch.float32, device=device)
        self.fluc_inp = torch.zeros(in_dim, dtype=torch.float32, device=device)
        self.nsamples = 0

    def add(self, inp):
        # [1, seq, dim] -> [dim, seq]
        values = inp.reshape(-1, inp.shape[-1]).t().to(torch.float32)
        n = self.nsamples
        self.baseline_inp *= n / (n + 1)
        self.baseline_inp += values.mean(dim=1) / (n + 1)
        if n == 0:
            # 原典は1本目で fluc を 0 に置き直す。1本目は寄与しない
            self.fluc_inp.zero_()
        else:
            # 原典の ``old_baseline_inp`` は **別名であって複製ではない**。
            # 直後の ``*=`` / ``+=`` が in-place なので、掛け合わせる2つの因子は
            # どちらも更新後の平均になる — つまり実体は偏差の2乗和である。
            # ここを clone にすると別の統計量になるので、別名のままにしてある
            deviation = values - self.baseline_inp.unsqueeze(1)
            self.fluc_inp *= (n - 1) / n
            self.fluc_inp += torch.sum(deviation * deviation, dim=1) / (n + 1)
        self.nsamples += 1


def _wifv(stats, linear):
    """WIFV = ゆらぎ × 出力側の重みの2乗和。"""
    return stats.fluc_inp * linear.weight.data.to(torch.float32).pow(2).sum(dim=0)


def _standardize(metric):
    """層ごと（dim=1）に標準化する。層をまたいで比べられるようにする段。"""
    return (metric - metric.mean(dim=1, keepdim=True)) / metric.std(dim=1, keepdim=True)


@torch.no_grad()
def collect(adapter, calibration, batch_chunk=64, progress=None):
    """全層の WIFV と補償用の平均入力を、刈る前のモデルの上で集める。"""
    shape = check_prunable(adapter)
    device = adapter.device
    hidden, mask, positions = capture_inputs(
        adapter, calibration.input_ids, chunk=batch_chunk)
    hidden = hidden.to(device)
    outs = torch.empty_like(hidden)

    attn_metric, mlp_metric = [], []
    attn_mean, mlp_mean = [], []
    for index in range(shape['n_layers']):
        layer = adapter.layers[index]
        targets = {'attn': layer.self_attn.o_proj, 'mlp': layer.mlp.down_proj}
        stats = {name: _BiasStats(module.in_features, device)
                 for name, module in targets.items()}
        handles = [
            module.register_forward_hook(
                lambda _module, inp, _out, name=name: stats[name].add(inp[0]))
            for name, module in targets.items()]
        try:
            for row in range(hidden.shape[0]):
                outs[row] = adapter.forward_layer(
                    index, hidden[row].unsqueeze(0), mask, positions)[0]
        finally:
            for handle in handles:
                handle.remove()

        attn_metric.append(_wifv(stats['attn'], targets['attn']).pow(2).cpu())
        mlp_metric.append(_wifv(stats['mlp'], targets['mlp']).cpu())
        attn_mean.append(stats['attn'].baseline_inp.cpu())
        mlp_mean.append(stats['mlp'].baseline_inp.cpu())
        hidden, outs = outs, hidden
        if progress is not None:
            progress(index)
    del hidden, outs
    torch.cuda.empty_cache()
    return {'attn_metric': torch.stack(attn_metric),
            'mlp_metric': torch.stack(mlp_metric),
            'attn_mean': attn_mean, 'mlp_mean': mlp_mean, 'shape': shape}


def _keep_at_least_one(mask, name, notes):
    """全滅した層を1本だけ残す。原典に無い歯止めだが、形が壊れるのを防ぐ。"""
    for index in (~mask.any(dim=1)).nonzero().view(-1).tolist():
        mask[index, 0] = True
        notes.append(f'層 {index} の {name} が全滅したので1本だけ残した')
    return mask


def build_plan(adapter, calibration, sparsity, scope='block', stats=None,
               batch_chunk=64, progress=None):
    """FLAP の計画を作る。モデルには触らない。

    ``scope='block'``: 原典の ``pruning_ratio``。attn+FFN 全体に対する比で、
    ヘッドと FFN ニューロンを1本のしきい値で同時に切る（AL-AM）。

    ``scope='mlp'``: attention を触らず、FFN ニューロン全体の ``sparsity`` を
    落とす。層をまたいだ標準化と大域しきい値はそのままなので、層別の配分規則
    としての性格は残る。提案手法とトークンあたりの活性パラメータが厳密に揃う。
    """
    if not 0 < sparsity < 1:
        raise ValueError(f'スパース率 {sparsity} は (0, 1) の外')
    if scope not in ('block', 'mlp'):
        raise ValueError(f'未知の scope {scope!r}（block / mlp から選ぶ）')
    if stats is None:
        stats = collect(adapter, calibration, batch_chunk=batch_chunk,
                        progress=progress)
    shape = stats['shape']
    n_layers, head_dim = shape['n_layers'], shape['head_dim']
    notes = []

    mlp_metric = _standardize(stats['mlp_metric'])
    attn_metric = _standardize(stats['attn_metric']).reshape(
        n_layers, -1, head_dim).mean(dim=2)

    if scope == 'block':
        # ヘッド1本 : FFN ニューロン1本 のパラメータ比 = 4·d·head_dim : 3·d
        weight_per_head = 4 * head_dim / 3.0
        flat = torch.cat([attn_metric.reshape(-1), mlp_metric.reshape(-1)])
        order = torch.argsort(flat, descending=True)
        # 並べ替えたあとの位置ごとの重み。前半 attn_metric.numel() 本がヘッド
        weights = torch.where(order < attn_metric.numel(),
                              torch.full_like(flat, weight_per_head),
                              torch.ones_like(flat))
        cumulative = torch.cumsum(weights, dim=0)
        target = weights.sum() * (1 - sparsity)
        threshold = flat[order][int(torch.argmin(torch.abs(cumulative - target)))]
        attn_mask = attn_metric > threshold
        mlp_mask = mlp_metric > threshold
    else:
        attn_mask = torch.ones_like(attn_metric, dtype=torch.bool)
        flat = mlp_metric.reshape(-1)
        n_keep = int(round(flat.numel() * (1 - sparsity)))
        threshold = torch.sort(flat, descending=True)[0][n_keep - 1]
        mlp_mask = mlp_metric >= threshold

    attn_mask = _keep_at_least_one(attn_mask, 'ヘッド', notes)
    mlp_mask = _keep_at_least_one(mlp_mask, 'FFN ニューロン', notes)

    layers = []
    for index in range(n_layers):
        heads = tuple(attn_mask[index].nonzero().view(-1).tolist())
        neurons = tuple(mlp_mask[index].nonzero().view(-1).tolist())
        layers.append(LayerPlan(
            layer=index,
            mlp_keep=neurons,
            head_keep=heads if scope == 'block' else None,
            mlp_bias=_bias(adapter.layers[index].mlp.down_proj,
                           stats['mlp_mean'][index], mlp_mask[index]),
            attn_bias=(_bias(adapter.layers[index].self_attn.o_proj,
                             stats['attn_mean'][index],
                             attn_mask[index].repeat_interleave(head_dim))
                       if scope == 'block' else None),
        ))
    return PrunePlan(
        method='flap', scope=scope, sparsity=sparsity, layers=tuple(layers),
        knobs={'metrics': 'WIFV', 'structure': 'AL-AM',
               'calib_samples': int(calibration.input_ids.shape[0]),
               'calib_seqlen': int(calibration.input_ids.shape[1])},
        notes=notes)


@torch.no_grad()
def _bias(linear, mean_inp, mask):
    """落とした入力チャネルの平均寄与を、出力側の bias に移す。"""
    device = linear.weight.device
    dropped = (mean_inp.to(torch.float32) * (~mask).to(torch.float32)).to(device)
    return (dropped @ linear.weight.data.to(torch.float32).t()).cpu()
