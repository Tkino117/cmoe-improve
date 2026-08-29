"""選択問題の train split から、**1問1系列**で引く校正セット。

**なぜあるか。** report/11 は校正データの中身を評価に近づけて `acc` を +0.036
動かした（dense との差の 54%）。だが近づけたのは中身だけで、詰め方は素の文章と
同じままだった — 1問を「文脈 + 正解の続き」1本にしたあと、タスク内で連結して
2048 トークンの窓を切る。窓の中では問題が何度もまたがり、attention は前の問題を
読む。そして活性プロファイル（``carve.profile``）は、その窓の**全位置を等しく**
数える。

評価の側で起きていることは違う。lm-eval は選択肢ごとに1問を独立に走らせ、
**続きのトークンの対数尤度だけ**を足して選択肢の点にする。文脈側の予測は1つも
採点に入らない。つまり正誤を決めているのは、1問の中のごく一部の位置である。

ここはその食い違いを両方とも埋める。

1. **1問1系列。** 他の問題は文脈に入らない。右詰めの padding だけを足す
2. **採点に効く位置に印を付ける。** ``TokenSet.segments`` に残し、読む側
   （``--profile-positions scored``）がそこだけを数える

**印は1つずれる。** 続きのトークンが位置 c..L-1 にあるとき、その対数確率を
出しているのは位置 c-1..L-2 の読み出しである。したがって印を付けるのは
``c-1..L-2`` で、続きのトークンが載っている位置そのものではない。数は同じ
（続きの長さ）で、区間が1つ手前へずれる。

**切れ目は lm-eval と同じ取り方にする。** ``encode_pair`` は
``lm_eval.api.model.TemplateLM._encode_pair`` の causal 枝と同じ手順である
（文脈の末尾の空白を続き側へ移し、全体を1回トークナイズしてから、文脈だけを
トークナイズした長さで切る）。校正が印を付ける位置と、評価が足す位置が、
同じ規則から出る。

**右詰めにするので attention マスクは要らない。** 因果 attention では実
トークンが自分より後ろを見ないので、埋めを右端に置くかぎり実トークンの活性は
埋めの有無に依らない。埋めの位置は自分では意味のない値を出すが、印が付かない
ので数えられない。

**量の数え方が benchtrain と違う。** ``n_samples × seqlen`` を、窓の総トークン
数ではなく**採点位置の総数**として読む（既定の n=8・seqlen=2048 なら 16,384
位置）。プロファイルの推定に実際に入るのがそこだけだからで、benchtrain の
16,384 トークンと揃うのはこの数である。系列の幅は ``MAX_TOKENS`` で決まる別物
である。
"""

from dataclasses import dataclass
from functools import lru_cache
import random

import torch

from cmoe.data.base import CONTEXT, PAD, SCORED, TokenSet, load_tokenizer
from cmoe.data.benchtrain import TASKS, allocate
from cmoe.data.harness import DEFAULT_CACHE, load_tasks, render_parts

# 1系列の上限。超えた問題は文脈の頭を落として収める（BOS は残す）。
# 全系列を最長に合わせて右詰めするので、上限がそのまま埋めの量を決める。
MAX_TOKENS = 256


@dataclass(frozen=True)
class Component:
    """1タスクぶんの引きの記録。``TokenSet.metadata()`` がそのまま書き出す。"""

    name: str
    documents: tuple
    scored_tokens: int
    context_tokens: int
    truncated: int
    pool_documents: int

    def metadata(self):
        return {
            'name': self.name,
            'n_questions': len(self.documents),
            'documents': list(self.documents),
            'pool_documents': self.pool_documents,
            'scored_tokens': self.scored_tokens,
            'context_tokens': self.context_tokens,
            'truncated': self.truncated,
            'split': 'train',
        }


@lru_cache(maxsize=None)
def documents(task_name, cache_dir=DEFAULT_CACHE):
    """1タスクの train split を (文脈, 続き) の対にしたもの。

    ``benchtrain.documents`` と同じ問題を、繋がずに持つ。繋いだものが要るのは
    窓を切る側だけで、こちらは切れ目を使う。
    """
    task = load_tasks([task_name], cache_dir=cache_dir)[task_name]
    if not task.has_training_docs():
        raise ValueError(f'{task_name} に train split が無い')
    return tuple(render_parts(task, doc) for doc in task.training_docs())


def encode_pair(tokenizer, context, continuation):
    """(全トークン, 続きの開始位置)。lm-eval の causal 枝と同じ切り方。

    文脈の末尾の空白を続き側へ移すのは、単語境界のトークン化を保つためで、
    ``TemplateLM._encode_pair`` がやっているのと同じ処理である。全体を1回
    トークナイズしてから文脈の長さで切るので、繋ぎ目でトークンが割れても
    校正と評価で同じ割れ方になる。
    """
    if not context:
        raise ValueError('文脈が空の問題は扱わない')
    spaces = len(context) - len(context.rstrip())
    if spaces > 0:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    whole = tokenizer(context + continuation, return_tensors='pt').input_ids[0]
    prefix = tokenizer(context, return_tensors='pt').input_ids[0]
    return whole, prefix.shape[0]


def build_row(ids, boundary, max_tokens=MAX_TOKENS):
    """1問を (トークン, 印) にする。長すぎる問題は文脈の頭を落とす。

    印は ``boundary - 1 .. len(ids) - 2``。続きの長さと同じ数だけ立ち、続きの
    トークンが載っている位置より1つ手前にある。

    落とすのは文脈の**中**からで、先頭の1トークン（Llama なら BOS）は残す。
    頭ごと落とすと、モデルが1度も見たことのない始まり方の系列になる。
    """
    length = ids.shape[0]
    if not 1 <= boundary < length:
        raise ValueError(
            f'続きの開始位置 {boundary} が 1..{length - 1} の外 — '
            '文脈か続きのどちらかがトークンを持たない')
    truncated = False
    if length > max_tokens:
        drop = length - max_tokens
        if boundary - drop < 1:
            raise ValueError(
                f'{length} トークンの問題を {max_tokens} に収めると文脈が'
                '残らない。MAX_TOKENS を上げるか、この問題を落とすこと')
        ids = torch.cat([ids[:1], ids[1 + drop:]])
        boundary -= drop
        length = max_tokens
        truncated = True
    marks = torch.full((length,), CONTEXT, dtype=torch.int8)
    marks[boundary - 1:length - 1] = SCORED
    return ids, marks, truncated


def draw_component(tokenizer, parts, budget, seed, name, max_tokens=MAX_TOKENS):
    """1タスクから、採点位置が budget に届くまで問題を引く。

    タスクごとに独立の種を使う（benchtrain と同じ ``f'{seed}:{name}'``）。
    budget をまたいだ問題は切らずに入れるので、実際の採点位置は budget を
    わずかに超える。超えた量は ``Component`` に残る。
    """
    order = list(range(len(parts)))
    random.Random(f'{seed}:{name}').shuffle(order)

    rows, marks, picked = [], [], []
    scored = context = truncated = 0
    for index in order:
        if scored >= budget:
            break
        ids, boundary = encode_pair(tokenizer, *parts[index])
        row, mark, was_truncated = build_row(ids, boundary, max_tokens)
        rows.append(row)
        marks.append(mark)
        picked.append(index)
        scored += int((mark == SCORED).sum())
        context += int((mark == CONTEXT).sum())
        truncated += int(was_truncated)
    if scored < budget:
        raise ValueError(
            f'{name} は train 全 {len(parts)} 問で採点位置 {scored} しか無い。'
            f'{budget} には届かない')
    return rows, marks, Component(
        name=name, documents=tuple(picked), scored_tokens=scored,
        context_tokens=context, truncated=truncated, pool_documents=len(parts))


def pack(rows, marks, pad_id):
    """可変長の行を [n, 幅] に右詰めする。幅は最長の行に合わせる。"""
    width = max(row.shape[0] for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    segments = torch.full((len(rows), width), PAD, dtype=torch.int8)
    for index, (row, mark) in enumerate(zip(rows, marks)):
        input_ids[index, :row.shape[0]] = row
        segments[index, :mark.shape[0]] = mark
    return input_ids, segments


def pad_token_id(tokenizer):
    """埋めに使う id。値は活性に効かない（右詰め + 因果 attention）。"""
    for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return candidate
    return 0


def calibration(model, seqlen, n_samples, seed, name='benchqa',
                cache_dir=DEFAULT_CACHE, tasks=TASKS, max_tokens=MAX_TOKENS):
    """``tasks`` を層化して引いた1問1系列のセット。既定は5タスク全部。

    ``n_samples × seqlen`` は**採点位置の総数**として読む（このモジュールの
    冒頭を見ること）。それをタスクへ等しく配るのは、ベンチの集計がタスクを
    等しく重み付けるマクロ平均だからで、``benchtrain.allocate`` が窓の本数で
    守っていた規則を、実質の量へ翻訳したものである。

    ``starts`` は空にする。窓を切っていないので、開始位置というものが無い。
    """
    budget = n_samples * seqlen
    if budget < len(tasks):
        raise ValueError(f'採点位置 {budget} では {len(tasks)} タスクに配れない')
    # 端数はタスクの並びの順に1つずつ。allocate と同じ配り方をここでも使う
    quota = allocate(budget, tasks)
    tokenizer = load_tokenizer(model)

    rows, marks, components = [], [], []
    for task_name in tasks:
        task_rows, task_marks, component = draw_component(
            tokenizer, documents(task_name, cache_dir), quota[task_name],
            seed, task_name, max_tokens)
        rows.extend(task_rows)
        marks.extend(task_marks)
        components.append(component)

    input_ids, segments = pack(rows, marks, pad_token_id(tokenizer))
    return TokenSet(f'{name}-train-qa', input_ids, (), tuple(components),
                    segments)
