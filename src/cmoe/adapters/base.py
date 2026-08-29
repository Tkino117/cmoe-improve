"""[軸1] モデル差異の境界。

上位のコードがモデルについて知ってよいのは、この2つだけである。

* ``DenseFFN``  … 層 ℓ の dense FFN を正規形で見たもの
* ``ModelAdapter`` … 層の並び、層の前半分(attention)の進め方、FFN の差し替え、
  最終段(norm + lm_head)

projection の名前、attention の呼び出し規約、層リストへの経路、seqlen の決め方
といった機種ごとの事情は、すべてアダプタの実装側に閉じる。
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


def slice_batch(value, start, stop, batch_size):
    """バッチ次元を持つものだけを切る。

    ``attention_mask`` は sdpa で None、eager で [bsz, 1, seq, seq]、
    ``position_ids`` はふつう [1, seq] でバッチ方向へ broadcast される。本当に
    バッチである先頭次元しか切ってはいけない — broadcast の側を切ると、その塊に
    別の塊の位置を黙って渡すことになる。
    """
    if isinstance(value, torch.Tensor) and value.dim() and value.shape[0] == batch_size:
        return value[start:stop]
    return value


@dataclass(frozen=True)
class DenseFFN:
    """一層分の dense FFN。carve と router はこれしか見ない。

    3つの Linear は元のモジュールそのもの（コピーではない）。重みを読むだけの
    利用を想定しており、書き換えてはならない。
    """

    hidden_size: int
    intermediate_size: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    act_fn: object


@dataclass
class LayerInputs:
    """層0 に入る隠れ状態と、以降の層が forward に必要とする付随情報。

    input_ids をモデルに通して層0 の直前で捕まえたもの。``attention_mask`` と
    ``position_ids`` の中身はモデルごとに違ってよく、アダプタの
    ``forward_attention`` / ``forward_layer`` にそのまま渡すためだけに持つ。
    """

    hidden: torch.Tensor
    attention_mask: object
    position_ids: object


class ModelAdapter(Protocol):
    """1モデルを、層の列として上位に見せる。"""

    model: object
    seqlen: int
    device: torch.device

    @property
    def n_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    def dense_ffn(self, index: int) -> DenseFFN:
        """層 index の dense FFN。すでに変換済みなら例外。"""

    def replace_ffn(self, index: int, module: nn.Module) -> None:
        """層 index の FFN を差し替える。モデルを変える唯一の操作。"""

    def is_converted(self, index: int) -> bool: ...

    def capture_layer_inputs(self, input_ids: torch.Tensor) -> LayerInputs:
        """input_ids を流し、層0 の入力を捕まえて返す。"""

    def forward_attention(self, index, hidden, attention_mask, position_ids):
        """層 index の attention 側だけを進め、(z, residual) を返す。

        z は FFN の入力（post-attention の正規化出力）、residual は
        層の出力が ``residual + ffn(z)`` になる側。
        """

    def forward_layer(self, index, hidden, attention_mask, position_ids):
        """層 index を丸ごと進める。"""

    def forward_suffix(self, start_layer, hidden, inputs, batch_chunk=None,
                       keep=None):
        """層 start_layer 以降を走らせて logits にする。

        ``start_layer == n_layers`` は空の suffix（最終段だけ）。層の呼び出し
        規約もバッチの切り方もモデルごとの事情なので、上位はこれ1つで済ませる。

        ``keep``（[bsz, seq] の bool）を渡すと、印の付いた位置の読み出しだけを
        [位置, 語彙] で返す。捨てる位置の logits を一度も並べないための引数で
        ある — [bsz, seq, 語彙] は語彙ぶんの幅があり、読まない位置まで持つと
        そこが一番大きい確保になる。
        """

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """最終 norm と lm_head を当てて logits にする。"""
