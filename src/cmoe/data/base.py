"""[軸2] データセット差異の境界。

上位が受け取るのは ``TokenSet`` だけである。どのデータセットをどう切り出したか
はここに閉じ、トークンのハッシュを一緒に持たせる。系が違えば数字が違うのが
このプロジェクトの一貫した経験なので、「どのトークンで測ったか」は結果と同じ
重みで記録する。
"""

from dataclasses import dataclass
import hashlib

import torch


def tensor_hash(tensor):
    """テンソルの内容そのもののハッシュ。形と dtype も混ぜる。"""
    value = tensor.detach().to('cpu').contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenSet:
    """ひと揃いのトークン列。

    name:      どこから取ったかの名前（'wikitext2-train-carve' など）
    input_ids: キャリブレーション用は [n_sequences, seqlen]、
               評価用は [1, tokens] の連結列
    starts:    切り出し開始位置。連続切りのときは空でよい
    """

    name: str
    input_ids: torch.Tensor
    starts: tuple = ()

    @property
    def n_sequences(self):
        return self.input_ids.shape[0]

    @property
    def n_tokens(self):
        return self.input_ids.numel()

    def metadata(self):
        return {
            'name': self.name,
            'shape': list(self.input_ids.shape),
            'n_tokens': self.n_tokens,
            'starts': list(self.starts),
            'token_hash': tensor_hash(self.input_ids),
        }


def load_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, use_fast=False)
