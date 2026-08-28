"""ExpertWeaver の多タスク校正セット（Flan）を、この土俵に載せ直したもの。

**なぜあるか。** 「校正データを測る対象に近づけると効く」（report/11・12）の
比較対象として、素の文章（wikitext2 / c4）は弱すぎる。この軸での既存の最良は
ExpertWeaver（arXiv:2602.15521）が使う **Flan の多タスク校正**であり、そこに
勝つ／並ぶことを見せないと「多タスク校正で既に足りている」で終わる。ここは
その行を作るためだけにある。

**元の設定。** ExpertWeaver は Appendix H で「Flan-v2 から 10 クラスタ・48
タスクを選び、各タスク5本の few-shot 例、計 240 サンプル」と書いている。ただし
論文の中で数が揃っていない — 本文 §2.2.2 は 48 タスク、§3 は 42 タスク、
Appendix G は「各タスク10サンプル」、そして Table 8 に実際に並んでいるのは 39
データセットである。ここが写すのは**実際に列挙されている Table 8 の 39**。

**入手の都合。** Flan-v2 のフルミックスはタスク名で引けない塊で配られている
ので、タスクごとに分かれている FLAN 2021 のミラー（``Muennighoff/flan``、
10テンプレート）から引く。Table 8 の 39 のうち、そこに無い6つ
（duorc / xsum / siqa / winogrande / race / winogender）は落とし、wmt14 の
en-de・en-ro は同じ言語対の wmt16 で代える。残る **36 タスク**がここの ``TASKS``
である。**WinoGrande が入らない**ことは結果の読みに効くので明記しておく —
評価5タスクのうち PIQA / HellaSwag / ARC-e / ARC-c はこの校正に入り、
WinoGrande だけが入らない。

**総量は他の行と同じ。** 引くのは n_samples × seqlen トークンで、benchtrain や
wikitext2 と1トークンも変わらない。ExpertWeaver の「240サンプル」は系列長が
書かれておらず量を写せないので、量ではなく**中身**を写す。

**タスクを交互に並べる。** 窓は 2048 トークンで本数は 8 しかないので、タスク
ごとに固めて連結すると1つの窓が1〜2タスクしか含まず、多タスク性が消える。
各タスクから採った例をラウンドロビンで交互に並べてから窓を切る。

引くのは train split だけで、評価に使う split は1問も入らない。
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import random
import urllib.request

import torch

from cmoe.data.base import TokenSet, load_tokenizer
from cmoe.data.benchtrain import CHARS_PER_TOKEN, MARGIN, MIN_MARGIN
from cmoe.data.wikitext2 import non_overlapping_starts, take

# ExpertWeaver Table 8 の並び（クラスタ順）。コメントは元の表記
TASKS = (
    # Reading Comprehension（duorc はミラーに無い）
    'squad_v1', 'squad_v2', 'drop', 'quac', 'record',
    # Summarization（xsum はミラーに無い）
    'cnn_dailymail', 'samsum', 'multi_news',
    # Translation（wmt14 の en-de / en-ro は wmt16 の同言語対で代替）
    'wmt14_enfr', 'wmt16_translate_deen', 'wmt16_translate_roen',
    # Commonsense Reasoning（siqa と winogrande はミラーに無い）
    'bool_q', 'piqa', 'cosmos_qa', 'hellaswag',
    # Natural Language Inference（anli は3ラウンドに分かれている）
    'mnli_matched', 'qnli', 'rte', 'wnli', 'anli_r1', 'anli_r2', 'anli_r3',
    # Coreference Resolution（winogender はミラーに無い）
    'wsc', 'definite_pronoun_resolution',
    # Sentiment Analysis
    'imdb_reviews', 'sentiment140', 'yelp_polarity_reviews',
    # Question Answering（race はミラーに無い。arc は easy/challenge に分かれる）
    'arc_easy', 'arc_challenge', 'openbookqa', 'trivia_qa',
    # Paraphrase Detection
    'glue_mrpc', 'glue_qqp',
    # Structure-to-Text
    'common_gen', 'e2e_nlg', 'dart',
)

URL = ('https://huggingface.co/datasets/Muennighoff/flan/resolve/main/'
       'train/{task}_10templates_train.jsonl')
DEFAULT_CACHE = os.environ.get(
    'CMOE_FLAN_CACHE', os.path.join('.cache', 'flan-v2'))
# 1タスクにつき先頭のこれだけを取る。要るのは1タスク数万文字で、multi_news は
# 385MB ある。行はテンプレート順に並んでいない（先頭数行が別テンプレート）ので、
# 先頭を取っても1つのテンプレートに偏らない
HEAD_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Component:
    """1タスクぶんの引きの記録。``TokenSet.metadata()`` がそのまま書き出す。"""

    name: str
    documents: tuple
    n_chars: int
    pool_documents: int

    def metadata(self):
        return {
            'name': self.name,
            'n_documents': len(self.documents),
            'documents': list(self.documents),
            'pool_documents': self.pool_documents,
            'n_chars': self.n_chars,
            'split': 'train',
        }


def shard(task, cache_dir=DEFAULT_CACHE):
    """タスク1つぶんの先頭を落として置く。2回目からはキャッシュを読む。"""
    path = os.path.join(cache_dir, f'{task}.jsonl')
    if os.path.exists(path):
        return path
    os.makedirs(cache_dir, exist_ok=True)
    request = urllib.request.Request(
        URL.format(task=task), headers={'Range': f'bytes=0-{HEAD_BYTES - 1}'})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    # 最後の行は途中で切れている。捨てる
    lines = payload.split(b'\n')[:-1]
    if not lines:
        raise ValueError(f'{task}: 先頭 {HEAD_BYTES} バイトに1行も無い')
    with open(path + '.partial', 'wb') as handle:
        handle.write(b'\n'.join(lines) + b'\n')
    os.replace(path + '.partial', path)
    return path


@lru_cache(maxsize=None)
def documents(task, cache_dir=DEFAULT_CACHE):
    """1タスクを、1サンプル1本のテキストに直したもの。

    FLAN の1行は ``inputs`` と ``targets`` に分かれている。繋ぎは改行1つに
    する（``inputs`` は問い・OPTIONS で終わるので、空白1つでは続きが同じ行に
    ぶら下がる）。
    """
    rows = []
    with open(shard(task, cache_dir)) as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(f'{row["inputs"]}\n{row["targets"]}')
    return tuple(rows)


def draw_task(texts, need, seed, name):
    """1タスクから need 文字ぶん。番号を返す。

    タスクごとに独立の種を使う（benchtrain と同じ理由）。``random.Random`` は
    文字列を sha512 で消費するので、種はプロセスに依存しない。
    """
    order = list(range(len(texts)))
    random.Random(f'{seed}:{name}').shuffle(order)
    picked, size = [], 0
    for index in order:
        picked.append(index)
        size += len(texts[index])
        if size >= need:
            break
    if size < need:
        raise ValueError(
            f'{name} は {len(texts)} 本 {size} 文字で、{need} 文字に届かない')
    return tuple(picked), size


def interleave(picked_by_task, tasks):
    """タスクをまたいでラウンドロビンに並べた (タスク, 番号) の列。"""
    order = []
    for position in range(max(len(picked_by_task[name]) for name in tasks)):
        for name in tasks:
            if position < len(picked_by_task[name]):
                order.append((name, picked_by_task[name][position]))
    return order


def calibration(model, seqlen, n_samples, seed, name='flanv2',
                cache_dir=DEFAULT_CACHE, tasks=TASKS):
    """36タスクを等量で交互に並べた列から、n_samples 本の窓を切る。"""
    tasks = tuple(tasks)
    need = n_samples * seqlen * CHARS_PER_TOKEN * MARGIN
    picked_by_task, components = {}, []
    for task in tasks:
        texts = documents(task, cache_dir)
        picked, n_chars = draw_task(texts, need // len(tasks), seed, task)
        picked_by_task[task] = picked
        components.append(Component(name=task, documents=picked,
                                    n_chars=n_chars, pool_documents=len(texts)))

    stream = '\n\n'.join(documents(task, cache_dir)[index]
                         for task, index in interleave(picked_by_task, tasks))
    tokenizer = load_tokenizer(model)
    ids = tokenizer(stream, return_tensors='pt').input_ids
    n_tokens = ids.shape[1]
    if n_tokens < n_samples * seqlen * MIN_MARGIN:
        raise ValueError(
            f'{n_tokens} トークンしか集まらなかった。'
            f'{n_samples}本×{seqlen} には最低 {n_samples * seqlen * MIN_MARGIN} 要る')

    starts = non_overlapping_starts(n_tokens, seqlen, n_samples, str(seed))
    return TokenSet(f'{name}-train-calib', take(ids, starts, seqlen),
                    tuple(starts), tuple(components))
