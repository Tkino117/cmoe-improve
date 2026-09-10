"""[対照] LLM-Pruner (Ma et al., NeurIPS 2023) の移送。**提案手法ではない。**

ExpertWeaver Table 2 の training-free 比較手法のひとつ。公式実装
(github.com/horseee/LLM-Pruner) の ``hf_prune.py --block_wise --pruner_type taylor
--taylor param_first`` を、この基盤のアダプタとデータ軸の上に移した。

**追加学習（LoRA による recovery tuning）は行わない。** 原典はプルーニング後に
alpaca で LoRA を当てるが、ここが比べているのは training-free の土俵であり、
ExpertWeaver Table 2 も追加学習なしの数字を並べている。原典の設計上この手法は
recovery を前提にしているので、そのぶん不利に出る（EW Table 2 の LLaMA3-8B
MMLU 24.2 はチャンスレベルである）。**手法の欠陥ではなく土俵の違いである。**

公式実装をそのまま持ち込めないのは、あちらが自前の ``modeling_llama.py`` と
同梱の ``torch_pruning`` に依存しており、このリポジトリの transformers 4.47.1 と
共存しないためである。移したのは重要度の式と刈り方だけで、依存グラフの探索は
要らない — Llama の block-wise では群が2種類しかなく、手で書ける。

**群と重要度（``taylor param_first`` / ``group_reduction sum``）**

* FFN 群（root = ``gate_proj`` の出力チャネル j）
  ``I[j] = Σ_i |W_gate[j,i]·G_gate[j,i]| + Σ_i |W_up[j,i]·G_up[j,i]|
           + Σ_o |W_down[o,j]·G_down[o,j]|``
* attention 群（root = ``q_proj`` の出力チャネル c、``consecutive_groups`` =
  head_dim なのでヘッド単位に潰れる）
  ``I[c] = Σ|W_q·G_q|[c,:] + Σ|W_k·G_k|[c,:] + Σ|W_v·G_v|[c,:] + Σ|W_o·G_o|[:,c]``
  をヘッド内で総和

勾配は10系列を**1バッチにまとめた1回の backward**から取る（原典と同じ）。
刈るのは層 ``[layer_start, layer_end)`` だけで、既定の 4〜30 は公式スクリプトの
値である。26/32 層しか刈らないので**名目のスパース率と実効の削減率は一致しない** —
``base.accounting`` が両方を出す。

原典は fp32 で勾配を取る（CPU 実行の既定）。ここは基盤に揃えて bf16 の
モデルで backward し、``|w·g|`` の総和だけ fp32 で持つ。48GB 1枚で
peak 25GiB に収まるのがこの形しかないためで、総和の側を fp32 にしてあるので
4096本の足し込みで桁が落ちることはない。
"""

import torch

from cmoe.prune.base import LayerPlan, PrunePlan, check_prunable

# 公式スクリプト scripts/llama_prune.sh の既定
DEFAULT_SAMPLES = 10
DEFAULT_SEQLEN = 64
DEFAULT_LAYER_START = 4
DEFAULT_LAYER_END = 30


def _salience(linear, dim):
    """``|W ⊙ ∇W|`` を dim 方向に潰す。fp32 で足す。"""
    weight = linear.weight
    if weight.grad is None:
        raise ValueError('勾配が無い。backward を通していない')
    return (weight.detach().to(torch.float32)
            * weight.grad.to(torch.float32)).abs().sum(dim=dim)


# 1回の backward に載せるトークン数の上限。原典の 10本×64 = 640 はこれを
# 下回るので、既定の校正では**分割が起きず1回の backward になる**（原典と同一）。
# 校正を提案手法（16本×2048 = 32k）に揃えるときだけ分割が要る — 32k トークンの
# 活性を全層ぶん保持すると数百GBになる
DEFAULT_TOKEN_CHUNK = 2048


def gradients(adapter, calibration, token_chunk=DEFAULT_TOKEN_CHUNK):
    """校正データで backward を通し、勾配をモデルに残す。

    原典は10系列を1つのバッチにまとめて1回だけ backward する。ここも既定では
    そうなるが、系列が長い校正では活性が載らないので塊に割る。

    **割っても勾配は1回でやるのと同じ値になる。** 求めたいのは全トークン平均の
    損失の勾配で、系列長が揃っているなら塊ごとの平均損失を「その塊が占める系列の
    割合」で重み付けして足したものに等しい。``loss * (この塊の本数 / 全本数)`` を
    backward すれば、勾配が加算されて厳密に一致する。
    """
    model = adapter.model
    adapter.to_device()
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.requires_grad_(True)
    ids = calibration.input_ids.to(adapter.device)
    total, seqlen = ids.shape
    step = max(1, token_chunk // seqlen)
    loss_sum = 0.0
    for start in range(0, total, step):
        chunk = ids[start:start + step]
        loss = model(chunk, labels=chunk).loss
        (loss * (chunk.shape[0] / total)).backward()
        loss_sum += float(loss.detach()) * chunk.shape[0] / total
    return loss_sum


def clear_gradients(model):
    """重要度を取ったら勾配を捨てる。13.5GB がそのまま残るので必須。"""
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.requires_grad_(False)
    torch.cuda.empty_cache()


def _mlp_importance(mlp):
    return (_salience(mlp.gate_proj, 1) + _salience(mlp.up_proj, 1)
            + _salience(mlp.down_proj, 0))


def _head_importance(attn, head_dim):
    per_channel = (_salience(attn.q_proj, 1) + _salience(attn.k_proj, 1)
                   + _salience(attn.v_proj, 1) + _salience(attn.o_proj, 0))
    return per_channel.view(-1, head_dim).sum(dim=1)


def _keep_lowest_removed(importance, n_pruned):
    """重要度の低い n_pruned 本を落とし、残りを昇順の番号で返す。"""
    if n_pruned <= 0:
        return tuple(range(importance.numel()))
    dropped = set(torch.argsort(importance)[:n_pruned].tolist())
    return tuple(index for index in range(importance.numel()) if index not in dropped)


def build_plan(adapter, calibration, sparsity, scope='block',
               layer_start=DEFAULT_LAYER_START, layer_end=DEFAULT_LAYER_END,
               token_chunk=DEFAULT_TOKEN_CHUNK, log=None):
    """LLM-Pruner の計画を作る。勾配は取るが、モデルの重みには触らない。

    ``scope='block'``: 原典の ``--pruning_ratio``。刈る層の attention ヘッドと
    FFN ニューロンをそれぞれ ``sparsity`` の割合だけ落とす（層ごとに一様＝local）。

    ``scope='mlp'``: attention を触らず、**モデル全体の FFN ニューロンの
    ``sparsity``** を落とす。刈る層が 26/32 しか無いので、層あたりの割合は
    ``sparsity × n_layers / 刈る層数`` に上がる。提案手法とトークンあたりの
    活性パラメータが厳密に揃う土俵。
    """
    if not 0 < sparsity < 1:
        raise ValueError(f'スパース率 {sparsity} は (0, 1) の外')
    if scope not in ('block', 'mlp'):
        raise ValueError(f'未知の scope {scope!r}（block / mlp から選ぶ）')
    shape = check_prunable(adapter)
    n_layers = shape['n_layers']
    layer_end = min(layer_end, n_layers)
    pruned_layers = list(range(layer_start, layer_end))
    if not pruned_layers:
        raise ValueError(f'刈る層が無い（{layer_start}〜{layer_end}）')

    if scope == 'mlp':
        ratio = sparsity * n_layers / len(pruned_layers)
        if ratio >= 1:
            raise ValueError(
                f'FFN ニューロン全体の {sparsity:.0%} を {len(pruned_layers)} 層だけで'
                f'落とすには層あたり {ratio:.0%} 要る。層範囲を広げる')
    else:
        ratio = sparsity

    loss = gradients(adapter, calibration, token_chunk=token_chunk)
    if log is not None:
        seqlen = calibration.input_ids.shape[1]
        step = max(1, token_chunk // seqlen)
        log(f'  backward loss={loss:.4f} '
            f'（{tuple(calibration.input_ids.shape)} を {step} 本ずつ）')

    intermediate, heads, head_dim = (
        shape['intermediate_size'], shape['n_heads'], shape['head_dim'])
    # 原典の n_pruned = C - int(C·(1-s))。int() は切り捨てなので、刈る側が1本多く
    # なることがある。ここも合わせる
    n_mlp_pruned = intermediate - int(intermediate * (1 - ratio))
    n_head_pruned = ((heads * head_dim - int(heads * head_dim * (1 - ratio)))
                     // head_dim)

    plans = []
    for index in pruned_layers:
        layer = adapter.layers[index]
        mlp_keep = _keep_lowest_removed(_mlp_importance(layer.mlp), n_mlp_pruned)
        head_keep = None
        if scope == 'block':
            head_keep = _keep_lowest_removed(
                _head_importance(layer.self_attn, head_dim), n_head_pruned)
        plans.append(LayerPlan(layer=index, mlp_keep=mlp_keep, head_keep=head_keep))
    clear_gradients(adapter.model)

    return PrunePlan(
        method='llm_pruner', scope=scope, sparsity=sparsity, layers=tuple(plans),
        knobs={'pruner_type': 'taylor', 'taylor': 'param_first',
               'group_reduction': 'sum', 'global_pruning': False,
               'channel_ratio': ratio,
               'layer_start': layer_start, 'layer_end': layer_end,
               'recovery_tuning': False,
               'calib_samples': int(calibration.input_ids.shape[0]),
               'calib_seqlen': int(calibration.input_ids.shape[1]),
               'backward_loss': loss},
        notes=[f'層 {layer_start}〜{layer_end - 1} のみを刈る'
               f'（{len(pruned_layers)}/{n_layers} 層。公式スクリプトの既定）',
               '追加学習（recovery tuning）は行わない'])
