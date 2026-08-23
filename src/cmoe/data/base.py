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
    components: 成分を混ぜたセットだけが持つ内訳。metadata() を持つものを
               並べる。単一ソースでは空

    starts は混合セットでは空にする。成分ごとに別の連結列を切るので、位置を
    1本に並べると何を基準にした値か分からなくなるためである。
    """

    name: str
    input_ids: torch.Tensor
    starts: tuple = ()
    components: tuple = ()

    @property
    def n_sequences(self):
        return self.input_ids.shape[0]

    @property
    def n_tokens(self):
        return self.input_ids.numel()

    def metadata(self):
        data = {
            'name': self.name,
            'shape': list(self.input_ids.shape),
            'n_tokens': self.n_tokens,
            'starts': list(self.starts),
            'token_hash': tensor_hash(self.input_ids),
        }
        if self.components:
            data['components'] = [component.metadata()
                                  for component in self.components]
        return data


@dataclass(frozen=True)
class Splits:
    """ルーターを作るのに要る3本のトークン列。

    carve:      分割と expert 重みを決める（既存の測定はすべて 8 系列）
    fit:        ルーター方式が代表を選ぶのに使う。carve と重ならない
    validation: 出来たルーターを診断するのに使う。訓練 split ですらない

    3本を分けるのは、代表を選んだデータでその代表を評価すると必ず良く見える
    ためである。carve と fit が重ならないことは引き方の側で保証する。
    """

    carve: TokenSet
    fit: TokenSet
    validation: TokenSet

    def metadata(self):
        return {
            'carve': self.carve.metadata(),
            'fit': self.fit.metadata(),
            'validation': self.validation.metadata(),
        }


def load_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, use_fast=False)
