"""選択問題の train split から、**1問 K 系列**（選択肢ごとに1本）で引く校正セット。

``benchqa`` の拡張である。あちらは1問を「文脈 + **正解の**続き」1本にし、採点に
効く位置に印を付けた。ここが足すのは**不正解肢**で、1問が選択肢の数だけの系列に
なる。

**なぜ不正解肢が要るか。** lm-eval が正誤を決めるのは K 本の対数尤度の比較で
あって、正解肢1本の値ではない。目的関数を「正解と、最も惜しい不正解の差」—
マージン — にするには、モデルに不正解肢も読ませて点を取らなければならない。
KL（``alloc.oracles.suffix_kl``）が親モデルとの近さしか見ないのに対し、
マージンは**採点そのもの**を見る。report/11・14 が繰り返し出したのは、その2つの
順位が一致しないという結果だった。

**予算は問題数で数える（既定）。** 問題の並び（``harness.calibration_docs``）
も、タスクごとの種（``f'{seed}:{name}'``）も、配り方（``benchtrain.allocate``）
も ``benchqa`` と同じものを通す。違うのは数える単位で、``benchqa`` が採点位置の
数を数えるところを、こちらはタスクごとの**問題数**を数える。

そうする理由は目的関数の形にある。位置ごとに閉じた量（KL）の平均なら、位置を
タスクへ等しく配るのが揃え方として正しい。マージンは**問題ごとの量**なので、
位置で配ると問題数が続きの長さに反比例して決まってしまう — 実際、正解肢の採点
位置を等しく配ると PIQA 11問に対して HellaSwag は7問にしかならず、マクロ平均の
1/5 を7問が担うことになる。

``budget_unit='gold_scored'`` を渡すと ``benchqa`` と同じ数え方に戻り、同じ引数で
まったく同じ問題が選ばれる（そこへ不正解肢が足される形になる）。report/14 と
校正の問題集合を揃えたまま目的関数だけを替えたいときはこちらである。

**したがってモデルに通すトークンは K 倍に増える。** 5タスクの選択肢数は
PIQA 2・WinoGrande 2・ARC-e/c 4〜5・HellaSwag 4 で、平均すると約3.2倍になる。
同じ問題数で目的関数を替えると、探索の前向きコストがそのぶん増える。**同じ
トークン数で揃えたい場合は問題数を減らすことになる** — どちらを揃えるかは実験の
側の判断なので、ここは問題数を揃える方に決めてある（マージンは問題ごとの量で、
問題数が減るとそのまま推定が粗くなる）。

**印の意味は benchqa と同じで、不正解肢にも付く。** 選択肢 j の系列では、
「その選択肢の続きの対数尤度を作っている位置」に ``SCORED`` が立つ。正解肢だけ
特別扱いしない — マージンは K 本すべての点を要る。1つずれる規約
（続きのトークン i の確率は位置 i-1 が出す）も同じである。

**行と問題の対応は ``ChoiceGroups`` が持つ。** どの行がどの問題のどの選択肢か、
正解番号はどれか、どのタスクから来たか。これが無いと、系列ごとの対数尤度を
問題に束ね直せない。
"""

from dataclasses import dataclass
from functools import lru_cache
import random

from cmoe.data.base import (CONTEXT, SCORED, ChoiceGroups, TokenSet,
                            load_tokenizer)
from cmoe.data.benchqa import (MAX_TOKENS, build_row, encode_pair, pack,
                               pad_token_id)
from cmoe.data.benchtrain import TASKS, allocate
from cmoe.data.harness import DEFAULT_CACHE, calibration_docs, render_choice_parts

# 予算の数え方。``questions`` はタスクごとに引く問題数、``gold_scored`` は正解肢
# の採点位置の数（``benchqa`` と同じ数え方で、同じ引数なら同じ問題が出る）
BUDGET_UNITS = ('questions', 'gold_scored')
DEFAULT_BUDGET_UNIT = 'questions'


@dataclass(frozen=True)
class Component:
    """1タスクぶんの引きの記録。``TokenSet.metadata()`` がそのまま書き出す。

    ``scored_tokens`` は K 本すべての採点位置の合計、``gold_scored_tokens`` は
    正解肢だけの合計である。予算に照らすのは後者で、``benchqa`` の
    ``scored_tokens`` と同じ数になる。
    """

    name: str
    documents: tuple
    n_rows: int
    scored_tokens: int
    gold_scored_tokens: int
    context_tokens: int
    truncated: int
    pool_documents: int

    def metadata(self):
        return {
            'name': self.name,
            'n_questions': len(self.documents),
            'documents': list(self.documents),
            'pool_documents': self.pool_documents,
            'n_rows': self.n_rows,
            'scored_tokens': self.scored_tokens,
            'gold_scored_tokens': self.gold_scored_tokens,
            'context_tokens': self.context_tokens,
            'truncated': self.truncated,
            'split': 'train',
        }


@lru_cache(maxsize=None)
def documents(task_name, cache_dir=DEFAULT_CACHE):
    """1タスクの校正用の問題を ``(選択肢ごとの (文脈, 続き), 正解番号)`` で返す。

    並びは ``benchqa.documents`` と同じ（どちらも ``calibration_docs`` をその
    まま流す）。同じ添字が同じ問題を指すので、種を揃えれば同じ問題が選ばれる。
    """
    return tuple(render_choice_parts(task, doc)
                 for task, doc in calibration_docs(task_name, cache_dir))


def build_question(tokenizer, parts, gold, max_tokens=MAX_TOKENS):
    """1問を、選択肢ごとの (トークン, 印) K 組にする。

    1本ずつは ``benchqa.build_row`` そのものである。長すぎる選択肢はその系列
    だけが文脈の頭を落とすので、K 本の文脈の長さが揃わないことがある。揃える
    必要は無い — 系列どうしは独立に走り、比べるのは採点位置の対数尤度の和だけ
    である。
    """
    rows, marks = [], []
    scored = gold_scored = context = truncated = 0
    for index, (prompt, continuation) in enumerate(parts):
        ids, boundary = encode_pair(tokenizer, prompt, continuation)
        row, mark, was_truncated = build_row(ids, boundary, max_tokens)
        rows.append(row)
        marks.append(mark)
        n_scored = int((mark == SCORED).sum())
        scored += n_scored
        if index == gold:
            gold_scored = n_scored
        context += int((mark == CONTEXT).sum())
        truncated += int(was_truncated)
    return rows, marks, scored, gold_scored, context, truncated


def draw_component(tokenizer, pool, budget, seed, name, max_tokens=MAX_TOKENS,
                   budget_unit=DEFAULT_BUDGET_UNIT):
    """1タスクから、予算に届くまで問題を引く。並びと種は benchqa と同じ。

    数え方が2つある。

    ``questions``     引いた**問題数**。マージンは問題ごとの量なので、これが
                      目的関数の推定の粗さを直接決める
    ``gold_scored``   **正解肢の**採点位置の数。``benchqa`` の数え方そのもので、
                      同じ引数なら同じ問題集合が出る

    既定が ``questions`` なのは、位置で数えるとタスクごとの問題数が続きの長さで
    決まってしまうからである（HellaSwag の続きは PIQA の3倍近くあるので、同じ
    位置数では問題数が 1/3 になる）。位置を揃えるのが正しいのは、目的関数が
    位置ごとに閉じた量の平均のときで、マージンはそうではない。
    """
    if budget_unit not in BUDGET_UNITS:
        raise ValueError(
            f'未知の予算の単位 {budget_unit!r}。{list(BUDGET_UNITS)} から選ぶ')
    order = list(range(len(pool)))
    random.Random(f'{seed}:{name}').shuffle(order)

    rows, marks, groups, picked = [], [], [], []
    scored = gold_scored = context = truncated = 0
    for index in order:
        drawn = len(picked) if budget_unit == 'questions' else gold_scored
        if drawn >= budget:
            break
        parts, gold = pool[index]
        (question_rows, question_marks, n_scored, n_gold_scored, n_context,
         n_truncated) = build_question(tokenizer, parts, gold, max_tokens)
        groups.append((tuple(range(len(rows), len(rows) + len(question_rows))),
                       gold))
        rows.extend(question_rows)
        marks.extend(question_marks)
        picked.append(index)
        scored += n_scored
        gold_scored += n_gold_scored
        context += n_context
        truncated += n_truncated
    drawn = len(picked) if budget_unit == 'questions' else gold_scored
    if drawn < budget:
        raise ValueError(
            f'{name} は train 全 {len(pool)} 問で {budget_unit} が {drawn} しか'
            f'無い。{budget} には届かない')
    return rows, marks, groups, Component(
        name=name, documents=tuple(picked), n_rows=len(rows),
        scored_tokens=scored, gold_scored_tokens=gold_scored,
        context_tokens=context, truncated=truncated, pool_documents=len(pool))


def calibration(model, seqlen, n_samples, seed, name='benchchoice',
                cache_dir=DEFAULT_CACHE, tasks=TASKS, max_tokens=MAX_TOKENS,
                budget_unit=DEFAULT_BUDGET_UNIT):
    """``tasks`` を層化して引いた、1問 K 系列のセット。既定は5タスク全部。

    ``n_samples × seqlen`` が予算で、単位は ``budget_unit`` が決める（既定は
    **問題数**）。タスクへ等しく配るのは、ベンチの集計がタスクを等しく重み付ける
    マクロ平均だからで、目的関数の側も同じ束ね方をする。

    ``gold_scored`` を選ぶと ``benchqa`` と同じ数え方になり、同じ引数でまったく
    同じ問題が選ばれる（そこへ不正解肢が足される形になる）。report/14 と校正の
    問題集合を揃えたまま目的関数だけを替えたいときはこちらである。

    ``starts`` は空にする。窓を切っていないので、開始位置というものが無い。
    """
    budget = n_samples * seqlen
    if budget < len(tasks):
        raise ValueError(f'予算 {budget} では {len(tasks)} タスクに配れない')
    quota = allocate(budget, tasks)
    tokenizer = load_tokenizer(model)

    rows, marks, components = [], [], []
    group_rows, group_gold, group_tasks = [], [], []
    for task_name in tasks:
        task_rows, task_marks, groups, component = draw_component(
            tokenizer, documents(task_name, cache_dir), quota[task_name],
            seed, task_name, max_tokens, budget_unit)
        # 行番号はタスクごとに 0 から数えてあるので、既に積んだぶんだけずらす
        offset = len(rows)
        for choices, gold in groups:
            group_rows.append(tuple(row + offset for row in choices))
            group_gold.append(gold)
            group_tasks.append(task_name)
        rows.extend(task_rows)
        marks.extend(task_marks)
        components.append(component)

    input_ids, segments = pack(rows, marks, pad_token_id(tokenizer))
    choices = ChoiceGroups(tuple(group_rows), tuple(group_gold),
                           tuple(group_tasks))
    choices.check_rows(input_ids.shape[0])
    return TokenSet(f'{name}-train-choice', input_ids, (), tuple(components),
                    segments, choices)
