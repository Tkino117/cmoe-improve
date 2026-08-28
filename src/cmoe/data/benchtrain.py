"""選択問題ベンチマークの train split から作る校正セット。

**なぜあるか。** 既存の校正セット（wikitext2 / c4 / slimpajama）はどれも素の
文章で、切り出した窓は文の途中から始まって単語の途中で終わり、2048 トークンの
あいだに話題が何度も変わる。一方で最後に測るのは選択問題で、そこでモデルが
読むのは数十トークンの独立した1問である。「校正データを測る対象に近づけると
何か動くか」を見るための系がこれである。

**評価に使う split には触らない。** 引くのは5タスクそれぞれの train split で、
lm-eval が採点に使う split（piqa/winogrande/hellaswag は validation、ARC は
test）は1問も入らない。既存のベンチ結果も dense の基準も、そのまま比較対象と
して生きる。

**成分ごとの本数を固定する。** slimpajama と同じ理由で、総量が n=8 と小さい
ため、タスクを区別せずに引くと seed ごとに組成が壊れる。配り方は「各タスクに
1本、余りは ``TASKS`` の並びの順に1本ずつ」で、seed には依らない。タスクを
等しく扱うのは、ベンチの集計が (seed × タスク) を層としたマクロ平均だからで
ある（問題数で重みを付けると、集計側の重み付けと食い違う）。

**1問は「文脈 + 正解の続き」1本にする。** 詰め方は ``harness.render_document``
にある。不正解の選択肢は入れない。

窓の切り方は slimpajama と同じで、タスク内で文書を連結してから
``non_overlapping_starts`` で引く。母集団が窓の総面積の数倍しか無い成分がある
（ARC-Challenge の train は 1,119 問しか無い）ので、重なりを許す
``draw_starts`` は使わない。
"""

from dataclasses import dataclass
from functools import lru_cache
import random

import torch

from cmoe.data.base import TokenSet, load_tokenizer
from cmoe.data.harness import DEFAULT_CACHE, load_tasks, render_document
from cmoe.data.wikitext2 import non_overlapping_starts, take

# ``eval.bench.DEFAULT_TASKS`` と同じ並び。ここで import しないのは依存の向き
# （data は eval を読まない）を守るためで、食い違いは
# ``tests/test_data_benchtrain.py`` が見る
TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')

# 連結する量。1トークン4文字として、窓の総面積の MARGIN 倍を集める
CHARS_PER_TOKEN = 4
MARGIN = 8
# 実際に集まったトークン数がこの倍率を割ったら断る。1トークン4文字という
# 見積りがタスクごとに外れても、黙って窓の密度が上がるのではなく止まる
MIN_MARGIN = 2


@dataclass(frozen=True)
class Component:
    """1タスクぶんの引きの記録。``TokenSet.metadata()`` がそのまま書き出す。"""

    name: str
    starts: tuple
    documents: tuple
    n_tokens: int
    n_chars: int
    pool_documents: int

    def metadata(self):
        return {
            'name': self.name,
            'n_sequences': len(self.starts),
            'starts': list(self.starts),
            'n_documents': len(self.documents),
            'documents': list(self.documents),
            'pool_documents': self.pool_documents,
            'pool_tokens': self.n_tokens,
            'chars_per_token': round(self.n_chars / self.n_tokens, 3),
            'split': 'train',
        }


def allocate(count, tasks=TASKS):
    """タスクごとの本数。各タスクに1本、余りは ``tasks`` の並びの順に1本ずつ。

    slimpajama の ``allocate`` と違って実比率で配らない。ベンチの集計が
    (seed × タスク) を層としたマクロ平均で、タスクを等しく重み付けるためである。
    順番で配るので、n が5の倍数でないときにどのタスクが厚くなるかは決まって
    いる（seed を振っても変わらない）。
    """
    tasks = tuple(tasks)
    if count < len(tasks):
        raise ValueError(f'{count}本では{len(tasks)}タスクに1本ずつ配れない')
    quota = {name: 1 for name in tasks}
    for index in range(count - len(tasks)):
        quota[tasks[index % len(tasks)]] += 1
    return quota


@lru_cache(maxsize=None)
def documents(task_name, cache_dir=DEFAULT_CACHE):
    """1タスクの train split を、1問1本のテキストに直したもの。

    ``run`` は seed × 配分のぶんだけ校正を読み直すので覚えておく。
    """
    task = load_tasks([task_name], cache_dir=cache_dir)[task_name]
    if not task.has_training_docs():
        raise ValueError(f'{task_name} に train split が無い')
    return tuple(render_document(task, doc) for doc in task.training_docs())


def _gather(texts, need, rng):
    """文書をシャッフルし、合計が need 文字に届くまで採る。番号を返す。"""
    order = list(range(len(texts)))
    rng.shuffle(order)
    picked = []
    size = 0
    for index in order:
        picked.append(index)
        size += len(texts[index])
        if size >= need:
            break
    return tuple(picked), size


def draw_component(tokenizer, texts, seqlen, count, seed, name):
    """1タスクから count 本。

    タスクごとに独立の種を使う。同じ seed を全タスクで使い回すと、タスクを
    またいで同じ開始位置の並びが出る。``random.Random`` は文字列を sha512 で
    消費するので、この種はプロセスや ``PYTHONHASHSEED`` に依存しない。
    """
    stream = f'{seed}:{name}'
    picked, n_chars = _gather(texts, count * seqlen * CHARS_PER_TOKEN * MARGIN,
                              random.Random(stream))

    ids = tokenizer('\n\n'.join(texts[index] for index in picked),
                    return_tensors='pt').input_ids
    n_tokens = ids.shape[1]
    if n_tokens < count * seqlen * MIN_MARGIN:
        raise ValueError(
            f'{name} は {len(picked)} 問 {n_chars} 文字で {n_tokens} トークン。'
            f'{count}本×{seqlen} には最低 {count * seqlen * MIN_MARGIN} 要る')

    starts = non_overlapping_starts(n_tokens, seqlen, count, stream)
    return take(ids, starts, seqlen), Component(
        name=name, starts=starts, documents=picked, n_tokens=n_tokens,
        n_chars=n_chars, pool_documents=len(texts))


def calibration(model, seqlen, n_samples, seed, name='benchtrain',
                cache_dir=DEFAULT_CACHE, tasks=TASKS):
    """``tasks`` を層化して引いた n_samples 本。既定は5タスク全部。

    ``tasks`` を1つに絞ると「そのタスクだけで校正する」系になる。総トークン数は
    n_samples × seqlen で変わらないので、タスクの数を減らしても量は動かない。
    5タスクぶんの窓を1タスクから引くことになるが、母集団が最も小さい
    ARC-Challenge（train 1,119 問・約 47,000 トークン）でも n=8 は
    ``MIN_MARGIN`` を満たす。

    ``TokenSet.starts`` は空にする。開始位置はタスクごとに別の連結列に対する
    ものなので、1本に並べると基準を失う。位置は ``components`` の側に残る。
    """
    counts = allocate(n_samples, tasks)
    tokenizer = load_tokenizer(model)
    blocks, components = [], []
    for task_name in tasks:
        block, component = draw_component(
            tokenizer, documents(task_name, cache_dir), seqlen,
            counts[task_name], seed, task_name)
        blocks.append(block)
        components.append(component)
    return TokenSet(f'{name}-train-calib', torch.cat(blocks, dim=0), (),
                    tuple(components))
