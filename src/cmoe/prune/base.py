"""[対照] 静的な構造化プルーニングの共通部分。**提案手法ではない。**

ExpertWeaver Table 2 が並べている training-free の比較手法（FLAP・LLM-Pruner）を、
この基盤の評価経路に載せるためだけの層である。配分の対照（``alloc/ew_rule.py`` など）
と違って、こちらは MoE 変換を通らない — FFN ニューロンと attention ヘッドを
**恒久的に削除**して dense のまま測る。

したがって MoE 変換と揃うのは**トークンあたりの活性パラメータ（＝ FLOPs）だけ**で
あり、メモリは揃わない。MoE 変換は全パラメータを保持したまま実行だけを疎にするが、
静的プルーニングは重みごと消す。表には両方の意味でのスパース率を書く必要がある。

スパース率の数え方も原典どうしで食い違う。この基盤の「25%」は **FFN ニューロンの
25%がトークンごとに不活性**（N=8 / A=6）であり、ブロック全体では 16.71% にしかならない。
一方 FLAP の ``pruning_ratio`` は attn+FFN 全体に対する比である。同じ「25%」と書くと
対照だけが 1.5 倍削ることになるので、``scope`` の2値でどちらの土俵かを明示する。

* ``block``   … 原典の定義そのまま（attn+FFN 全体の比）。ExpertWeaver Table 2 と同じ土俵
* ``mlp``     … FFN だけを刈り、削る FFN ニューロンの割合を提案手法に揃える。
                トークンあたりの活性パラメータが厳密に一致する土俵

``PrunePlan`` は「どの層のどの構造を残すか」だけを持ち、モデルには触らない。
決めることと当てることを分けてあるのは、``summary.json`` に残すのが計画の側で、
当てた結果は評価の数字にしか現れないからである。
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerPlan:
    """1層分の「残す構造」。``None`` はその側を触らないことを表す。

    ``mlp_bias`` / ``attn_bias`` は FLAP の補償項で、落とした入力チャネルの
    平均寄与を出力側の bias に移したもの。LLM-Pruner は補償を持たないので None。
    """

    layer: int
    mlp_keep: tuple = None
    head_keep: tuple = None
    mlp_bias: object = None
    attn_bias: object = None

    def as_dict(self):
        return {
            'layer': self.layer,
            'n_mlp_kept': None if self.mlp_keep is None else len(self.mlp_keep),
            'n_heads_kept': None if self.head_keep is None else len(self.head_keep),
            'mlp_keep': None if self.mlp_keep is None else list(self.mlp_keep),
            'head_keep': None if self.head_keep is None else list(self.head_keep),
            'has_bias': self.mlp_bias is not None or self.attn_bias is not None,
        }


@dataclass
class PrunePlan:
    """1モデル分の計画と、その場で数えたパラメータ会計。"""

    method: str
    scope: str
    sparsity: float
    layers: tuple
    knobs: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {
            'method': self.method,
            'scope': self.scope,
            'sparsity': self.sparsity,
            'knobs': dict(self.knobs),
            'notes': list(self.notes),
            'layers': [row.as_dict() for row in self.layers],
        }


def model_shape(adapter):
    """会計に要る形。アダプタの向こう側を覗くのはここだけにする。"""
    config = adapter.model.config
    hidden = config.hidden_size
    heads = config.num_attention_heads
    return {
        'n_layers': adapter.n_layers,
        'hidden_size': hidden,
        'intermediate_size': config.intermediate_size,
        'n_heads': heads,
        'head_dim': getattr(config, 'head_dim', hidden // heads),
        'n_kv_heads': config.num_key_value_heads,
    }


def block_params(shape):
    """1ブロックの FFN と attention のパラメータ数。埋め込みと lm_head は数えない。

    FLAP の ``check_sparsity`` / ``cal_remove_neuron`` と同じ数え方である
    （FFN = 3·d·i、attention = 4·d²）。ここを揃えないと「25%」が別のものになる。
    """
    d, i = shape['hidden_size'], shape['intermediate_size']
    return {'ffn': 3 * d * i, 'attn': 4 * d * d, 'total': 3 * d * i + 4 * d * d}


def moe_activated_sparsity(shape, n_experts, n_active):
    """MoE 変換（N/A）が、ブロック全体の何割を不活性にしているか。

    提案手法の 25%（N=8 / A=6）は FFN ニューロンの比であって、ブロック全体では
    16.71% である。対照を同じ土俵に置くための基準がこれ。
    """
    parts = block_params(shape)
    return (1 - n_active / n_experts) * parts['ffn'] / parts['total']


def accounting(plan, shape):
    """計画が実際に削るパラメータを数える。名目比と実効比の両方を返す。

    LLM-Pruner のように一部の層しか刈らない手法では、名目のスパース率と実効の
    削減率が一致しない。表に両方載せるための数字をここで作る。
    """
    parts = block_params(shape)
    n_layers = shape['n_layers']
    d, i = shape['hidden_size'], shape['intermediate_size']
    heads, head_dim = shape['n_heads'], shape['head_dim']

    # 計画に載っていない層は手つかず。LLM-Pruner は 26/32 層しか刈らないので、
    # ここを計画の長さで数えると分母が縮んで実効の削減率が見かけ上あがる
    by_layer = {row.layer: row for row in plan.layers}
    removed_ffn = removed_attn = 0
    kept_mlp, kept_heads = [], []
    for index in range(n_layers):
        row = by_layer.get(index)
        n_mlp = i if row is None or row.mlp_keep is None else len(row.mlp_keep)
        n_head = heads if row is None or row.head_keep is None else len(row.head_keep)
        removed_ffn += 3 * d * (i - n_mlp)
        removed_attn += 4 * d * head_dim * (heads - n_head)
        kept_mlp.append(n_mlp)
        kept_heads.append(n_head)

    total = n_layers * parts['total']
    return {
        'removed_ffn_params': removed_ffn,
        'removed_attn_params': removed_attn,
        'block_params': total,
        # 表の共通軸。全手法をここで並べる
        'effective_block_sparsity': (removed_ffn + removed_attn) / total,
        # FFN ニューロンのうち何割が消えたか。提案手法の「25%」と同じ数え方
        'ffn_neuron_sparsity': removed_ffn / (n_layers * parts['ffn']),
        'mean_intermediate': sum(kept_mlp) / n_layers,
        'mean_heads': sum(kept_heads) / n_layers,
        'intermediate_per_layer': kept_mlp,
        'heads_per_layer': kept_heads,
    }


@torch.no_grad()
def apply_plan(adapter, plan):
    """計画をモデルに当てる。**破壊的**で、元に戻す道は無い（読み直す）。

    transformers 4.47.1 の ``LlamaAttention`` は ``num_heads`` / ``head_dim`` を
    インスタンス属性で持つので、層ごとにヘッド数が違ってよい。``nn.Linear`` の
    bias は ``F.linear`` がそのまま使うので、config が bias 無しでも後から生やせる。
    どちらもこの基盤の PPL 経路・lm-eval 経路で動くことを確かめてある。
    """
    for row in plan.layers:
        layer = adapter.layers[row.layer]
        if row.mlp_keep is not None:
            _prune_mlp(layer.mlp, row.mlp_keep, row.mlp_bias)
        if row.head_keep is not None:
            _prune_attention(layer.self_attn, row.head_keep, row.attn_bias)
    return adapter


def _index(keep, device):
    return torch.as_tensor(list(keep), dtype=torch.long, device=device)


def _prune_mlp(mlp, keep, bias):
    device = mlp.down_proj.weight.device
    index = _index(keep, device)
    mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[index].contiguous()
    mlp.up_proj.weight.data = mlp.up_proj.weight.data[index].contiguous()
    mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, index].contiguous()
    mlp.gate_proj.out_features = mlp.up_proj.out_features = len(keep)
    mlp.down_proj.in_features = len(keep)
    mlp.intermediate_size = len(keep)
    if bias is not None:
        mlp.down_proj.bias = nn.Parameter(
            bias.to(device=device, dtype=mlp.down_proj.weight.dtype))


def _prune_attention(attn, keep, bias):
    device = attn.o_proj.weight.device
    head_dim = attn.head_dim
    heads = _index(keep, device)
    columns = (heads.unsqueeze(1) * head_dim
               + torch.arange(head_dim, device=device)).reshape(-1)
    for name in ('q_proj', 'k_proj', 'v_proj'):
        proj = getattr(attn, name)
        proj.weight.data = proj.weight.data[columns].contiguous()
        proj.out_features = len(columns)
    attn.o_proj.weight.data = attn.o_proj.weight.data[:, columns].contiguous()
    attn.o_proj.in_features = len(columns)
    # Llama-2 は GQA を使わない（kv ヘッドと q ヘッドが同数）。GQA のモデルでは
    # q と kv を同じマスクで刈れないので、build_plan の側で断っている
    attn.num_heads = len(keep)
    attn.num_key_value_heads = len(keep)
    attn.num_key_value_groups = 1
    attn.hidden_size = len(keep) * head_dim
    if bias is not None:
        attn.o_proj.bias = nn.Parameter(
            bias.to(device=device, dtype=attn.o_proj.weight.dtype))


def check_prunable(adapter):
    """刈れる形かどうかを、計画を作る前に確かめる。"""
    shape = model_shape(adapter)
    if shape['n_kv_heads'] != shape['n_heads']:
        raise ValueError(
            f'kv ヘッド {shape["n_kv_heads"]} と q ヘッド {shape["n_heads"]} が'
            '違う（GQA）。FLAP も LLM-Pruner も q/k/v/o を同じヘッドマスクで刈るので、'
            'このままでは移送できない')
    return shape


def capture_inputs(adapter, input_ids, chunk=64):
    """層0 の入力を、系列を分けて捕まえる。

    ``capture_layer_inputs`` は埋め込みだけを走らせて層0 の直前で止まるが、
    2048系列を一度に通すと [2048, seq, 4096] を一度に確保する。刈る前のモデルは
    13.5GB あるので、ここは分けて積む。
    """
    hidden, mask, positions = [], None, None
    for start in range(0, input_ids.shape[0], chunk):
        captured = adapter.capture_layer_inputs(input_ids[start:start + chunk])
        hidden.append(captured.hidden)
        if mask is None:
            mask, positions = captured.attention_mask, captured.position_ids
        elif captured.attention_mask is not None:
            # padding のある系列では塊ごとにマスクが違う。この経路は素の文章を
            # 窓で切る校正しか想定していないので、違いが出たら黙って進まない
            if not torch.equal(captured.attention_mask, mask):
                raise ValueError('塊ごとに attention_mask が違う')
    return torch.cat(hidden, dim=0), mask, positions
