"""SlimPajama。7つの成分を**層化して**引く混合キャリブレーション。

wikitext2 と c4 は単一ソースで、「成分の内訳」という概念が無い。ここは1つの
校正セットの中に7成分（CommonCrawl / C4 / GitHub / ArXiv / Book / Wikipedia /
StackExchange）を持つ。

成分ごとの本数を固定するのが要点である。総量が n=8〜16 と小さいので、
成分を区別せずランダムに引くと **seed ごとに組成が壊れる**。何を測っても
「その seed がたまたま引いた組成」の効果と混ざり、report/01 で測った seed
ばらつきに新しい成分が乗るだけになる。

出所は ``DKYoon/SlimPajama-6B``（``cerebras/SlimPajama-627B`` の 10% サンプル。
本家 627B と ``togethercomputer/RedPajama-Data-1T-Sample`` は 2026-08 時点で
認証なしには引けない）。parquet 1シャードに7成分すべてが入っており、
``meta.redpajama_set_name`` が成分名を持つ。8〜16本を引くには1枚で足りるので、
c4 が ``TRAIN_FILE`` を1つ固定しているのと同じように1枚に固定する。

窓の切り方は wikitext2 と同じ「成分内で連結してから切る」である。c4 の
「seqlen 以上の文書だけ採る」は使えない。1シャードの文書長の中央値は C4 が
1,235文字・Wikipedia が 1,128文字で、2048トークンに届く文書は C4 で 3.9% しか
無く、そこだけを採ると極端な長文書バイアスが乗る。

ただし窓は ``non_overlapping_starts`` で引く。成分ごとに必要量だけを連結する
ので母集団が小さく、wikitext2 と同じ ``draw_starts`` を使うと窓どうしが高い
確率で重なる（n=16 の CommonCrawl で 78%、重複トークン約10%。wikitext2 の
n=16 は 0.7%）。校正データの中身を比べたいときに、窓の冗長度という別の差を
混ぜ込まないための選択である。

**n をまたいだ比較はできない。** 連結する量が成分の本数に比例するので、
本数が変わった成分は文書集合ごと総入れ替えになる一方、本数が変わらない成分
（n=8→16 の ArXiv など）は同一の窓のまま残る。wikitext2 のように「n を
増やしても先頭 n 本は変わらない」形にはならない。

pyarrow と huggingface_hub は ``datasets==2.21.0`` が必ず連れてくるので、
pyproject には直接は書いていない。ダウンロード先は ``hf_hub_download`` の
既定（``HF_HOME`` / ``HF_HUB_CACHE``）に任せる。README がベンチについて
``.cache/hf-datasets`` を分けている理由は ``datasets`` の索引バージョンなので、
hub から直接引くここは対象外である。
"""

from dataclasses import dataclass
from functools import lru_cache
import random

import torch

from cmoe.data.base import TokenSet, load_tokenizer
from cmoe.data.wikitext2 import non_overlapping_starts, take

REPO = 'DKYoon/SlimPajama-6B'
SHARD = 'data/train-00000-of-00048-ab2b35705f029d94.parquet'

# 並べる順。実比率の降順に置いてあるが、順序自体は**固定であること**にしか
# 意味が無い（連結の順が変わればトークン列が変わる）。
SUBSETS = ('RedPajamaCommonCrawl', 'RedPajamaC4', 'RedPajamaGithub',
           'RedPajamaArXiv', 'RedPajamaBook', 'RedPajamaWikipedia',
           'RedPajamaStackExchange')

# 連結する量。1トークン4文字として、窓の総面積の MARGIN 倍を集める。MARGIN は
# 窓の密度そのもので、大きいほど棄却が減り、集まる文書も増える。
CHARS_PER_TOKEN = 4
MARGIN = 8
# 実際に集まったトークン数がこの倍率を割ったら断る。1トークン4文字という
# 見積りが成分ごとに外れても（コードや LaTeX は文字あたりのトークンが多い）、
# 黙って密度が上がるのではなく止まるようにする。
MIN_MARGIN = 2


@dataclass(frozen=True)
class Component:
    """1成分ぶんの引きの記録。``TokenSet.metadata()`` がそのまま書き出す。"""

    name: str
    starts: tuple
    documents: tuple
    n_tokens: int
    n_chars: int

    def metadata(self):
        return {
            'name': self.name,
            'n_sequences': len(self.starts),
            'starts': list(self.starts),
            'documents': list(self.documents),
            'pool_tokens': self.n_tokens,
            'chars_per_token': round(self.n_chars / self.n_tokens, 3),
            'repo': REPO,
            'shard': SHARD,
        }


def allocate(shares, count):
    """成分ごとの本数。**各成分に最低1本**、残りを実比率の最大剰余法で配る。

    最低1本を先に配るのは、比率どおりに配ると小さい成分（ArXiv は 3.6%、
    StackExchange は 2.9%）が n=16 でも0本になり、「混ぜた」と言えなくなる
    ためである。残りを比率で配るので、総量を増やせば実比率に近づく。

    shares の値は正であれば絶対量でも割合でもよい（和で正規化する）。
    """
    names = list(shares)
    if count < len(names):
        raise ValueError(f'{count}本では{len(names)}成分に1本ずつ配れない')
    if any(value <= 0 for value in shares.values()):
        raise ValueError('比率に正でない値がある')
    total = sum(shares.values())

    quota = {name: 1 for name in names}
    rest = count - len(names)
    if rest:
        exact = {name: rest * shares[name] / total for name in names}
        extra = {name: int(exact[name]) for name in names}
        # 端数の大きい順。同点は名前順で決める（実行ごとに変わらないため）
        order = sorted(names, key=lambda name: (-(exact[name] - extra[name]), name))
        for name in order[:rest - sum(extra.values())]:
            extra[name] += 1
        for name in names:
            quota[name] += extra[name]
    return quota


@lru_cache(maxsize=1)
def _shard_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, SHARD, repo_type='dataset')


@lru_cache(maxsize=1)
def _documents(path):
    """成分名 -> 文書のタプル。SUBSETS の順に並べた dict を返す。

    292MB のシャードを Python 文字列へ展開するので、run が seed × 配分の
    ぶんだけ校正を読み直す使われ方（cli.command_run）に備えて覚えておく。
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=['text', 'meta'])
    texts = table.column('text').to_pylist()
    names = [row['redpajama_set_name'] for row in table.column('meta').to_pylist()]

    grouped = {name: [] for name in SUBSETS}
    for name, text in zip(names, texts):
        if name in grouped:
            grouped[name].append(text)
    missing = [name for name in SUBSETS if not grouped[name]]
    if missing:
        raise ValueError(f'{SHARD} に {missing} の文書が無い')
    return {name: tuple(rows) for name, rows in grouped.items()}


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
    """1成分から count 本。

    成分ごとに独立の種を使う。同じ seed を全成分で使い回すと、成分をまたいで
    同じ開始位置の並びが出る（成分の長さが違うので実害は小さいが、独立に
    引いたことにはならない）。``random.Random`` は文字列を sha512 で消費する
    ので、この種はプロセスや ``PYTHONHASHSEED`` に依存しない。
    """
    stream = f'{seed}:{name}'
    picked, n_chars = _gather(texts, count * seqlen * CHARS_PER_TOKEN * MARGIN,
                              random.Random(stream))

    ids = tokenizer('\n\n'.join(texts[index] for index in picked),
                    return_tensors='pt').input_ids
    n_tokens = ids.shape[1]
    if n_tokens < count * seqlen * MIN_MARGIN:
        raise ValueError(
            f'{name} は {len(picked)} 文書 {n_chars} 文字で {n_tokens} トークン。'
            f'{count}本×{seqlen} には最低 {count * seqlen * MIN_MARGIN} 要る')

    starts = non_overlapping_starts(n_tokens, seqlen, count, stream)
    return take(ids, starts, seqlen), Component(
        name=name, starts=starts, documents=picked,
        n_tokens=n_tokens, n_chars=n_chars)


def calibration(model, seqlen, n_samples, seed, name='slimpajama'):
    """7成分を層化して引いた n_samples 本。

    成分ごとの本数は、そのシャードで実測した文字数の比から ``allocate`` が
    決める。比率を定数で持たないのは、シャードを差し替えたときに定数だけが
    古いまま残るのを避けるためである。

    ``TokenSet.starts`` は空にする。開始位置は成分ごとに別の連結列に対する
    ものなので、1本に並べると基準を失う。位置は ``components`` の側に残る。
    """
    tokenizer = load_tokenizer(model)
    grouped = _documents(_shard_path())
    shares = {subset: sum(len(text) for text in grouped[subset])
              for subset in SUBSETS}
    quota = allocate(shares, n_samples)

    rows = []
    components = []
    for subset in SUBSETS:
        block, component = draw_component(
            tokenizer, grouped[subset], seqlen, quota[subset], seed, subset)
        rows.append(block)
        components.append(component)
    return TokenSet(f'{name}-train-calib', torch.cat(rows, dim=0),
                    components=tuple(components))
