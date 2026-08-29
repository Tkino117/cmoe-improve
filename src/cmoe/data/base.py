"""[軸2] データセット差異の境界。

上位が受け取るのは ``TokenSet`` だけである。どのデータセットをどう切り出したか
はここに閉じ、トークンのハッシュを一緒に持たせる。系が違えば数字が違うのが
このプロジェクトの一貫した経験なので、「どのトークンで測ったか」は結果と同じ
重みで記録する。
"""

from dataclasses import dataclass
import hashlib

import torch

# ``TokenSet.segments`` の値。位置ごとに「モデルがそこで何をしているか」を表す。
#
# ``SCORED`` は**採点される対数尤度を作る位置**である。lm-eval は選択肢の
# トークン i の対数確率を位置 i-1 の読み出しから取るので、続きが位置
# c..L-1 にあるなら、点を作っているのは位置 c-1..L-2 の活性であって、
# 続きのトークンが載っている位置そのものではない。1つずれる。
PAD = 0
CONTEXT = 1
SCORED = 2


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
    segments:  input_ids と同じ形の [n_sequences, seqlen] int8。位置ごとに
               PAD / CONTEXT / SCORED のどれか。**素の文章のセットは持たない**
               （None）。持つのは、1問1系列で引いて「どこが採点に効く位置か」
               を知っているセットだけである

    starts は混合セットでは空にする。成分ごとに別の連結列を切るので、位置を
    1本に並べると何を基準にした値か分からなくなるためである。
    """

    name: str
    input_ids: torch.Tensor
    starts: tuple = ()
    components: tuple = ()
    segments: object = None

    @property
    def n_sequences(self):
        return self.input_ids.shape[0]

    @property
    def n_tokens(self):
        return self.input_ids.numel()

    def scored_mask(self):
        """採点に効く位置だけ True の [n_sequences, seqlen]。無ければ None。"""
        if self.segments is None:
            return None
        return self.segments == SCORED

    def position_weights(self, scored_weight=None):
        """オラクルが位置ごとに掛ける重み [n_sequences, seqlen]。要らなければ None。

        ``scored_weight`` は**採点位置の群が持つ重みの割合** s である。位置1つ
        あたりの重みは、採点位置が s / (採点位置の数)、文脈位置が
        (1 - s) / (文脈位置の数)、埋めは常に 0 になる。群ごとに割るので、s は
        「答え部分が目的関数のどれだけを占めるか」そのものになる。

          s = 1      答え部分だけを見る
          s = 0      文脈だけを見る（対照。「位置の効果」の反対側）
          s = 採点位置の数 / 実位置の数    実位置を等しく見る（None と同じ順序）

        ``None`` は「実位置を等しく、埋めだけ 0」である。**印を持たないセット
        では None を返す** — 全位置が等しいので重みという概念が要らず、既存の
        すべての経路がそこを通る（1ビットも動かない）。

        返す重みは相対値でよい。読む側は重み付き平均を取るので、合計がいくつ
        であっても値は変わらない。
        """
        if self.segments is None:
            if scored_weight is not None:
                raise ValueError(
                    f'{self.name} は採点位置の印を持たないので、答え部分の'
                    '重みを決められない')
            return None
        scored = self.segments == SCORED
        context = self.segments == CONTEXT
        weights = torch.zeros(self.segments.shape, dtype=torch.float32)
        if scored_weight is None:
            weights[scored | context] = 1.0
            return weights
        if not 0.0 <= scored_weight <= 1.0:
            raise ValueError(f'答え部分の重みは 0..1（{scored_weight}）')
        n_scored, n_context = int(scored.sum()), int(context.sum())
        # 重みを持つ側の群が空なら、目的関数が空になる。黙ってもう一方だけの
        # 数を返さず断る
        if scored_weight > 0.0:
            if n_scored == 0:
                raise ValueError(f'{self.name} に採点位置が1つも無い')
            weights[scored] = scored_weight / n_scored
        if scored_weight < 1.0:
            if n_context == 0:
                raise ValueError(f'{self.name} に文脈位置が1つも無い')
            weights[context] = (1.0 - scored_weight) / n_context
        return weights

    def metadata(self):
        data = {
            'name': self.name,
            'shape': list(self.input_ids.shape),
            'n_tokens': self.n_tokens,
            'starts': list(self.starts),
            'token_hash': tensor_hash(self.input_ids),
        }
        if self.segments is not None:
            data['segments'] = {
                'pad': int((self.segments == PAD).sum()),
                'context': int((self.segments == CONTEXT).sum()),
                'scored': int((self.segments == SCORED).sum()),
                'hash': tensor_hash(self.segments),
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
