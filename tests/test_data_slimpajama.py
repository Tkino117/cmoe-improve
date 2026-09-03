"""層化した混合キャリブレーション。

配分規則（``allocate``）と引き方（``draw_component`` / ``_documents``）を
CPU だけで見る。トークナイザは偽物で置き換え、シャードは 10 行の parquet を
その場で書く。実物を 292MB 落とすところは
``CMOE_SLIMPAJAMA_INTEGRATION=1`` のときだけ走らせる（test_bench.py と同じ扱い）。
"""

import os
import random
from types import SimpleNamespace

import pytest
import torch

from cmoe.data.slimpajama import (MARGIN, MIN_MARGIN, SUBSETS, allocate,
                                  draw_component, _documents, _draw_set,
                                  _gather, _pool)
from cmoe.data.wikitext2 import intervals_overlap

INTEGRATION = os.environ.get('CMOE_SLIMPAJAMA_INTEGRATION') == '1'

# 1シャードで実測した文字数（成分の実比率）。規則の入力として使うだけで、
# 本番は毎回シャードから数え直す
MEASURED = {
    'RedPajamaCommonCrawl': 268.4,
    'RedPajamaC4': 140.6,
    'RedPajamaGithub': 22.4,
    'RedPajamaArXiv': 17.8,
    'RedPajamaBook': 16.9,
    'RedPajamaWikipedia': 12.6,
    'RedPajamaStackExchange': 14.1,
}

# 実物と同じ 16 本のときの内訳
EXPECTED_16 = {
    'RedPajamaCommonCrawl': 6,
    'RedPajamaC4': 4,
    'RedPajamaGithub': 2,
    'RedPajamaArXiv': 1,
    'RedPajamaBook': 1,
    'RedPajamaWikipedia': 1,
    'RedPajamaStackExchange': 1,
}


class FakeTokenizer:
    """1トークン = chars_per_token 文字。トークン id は連番。"""

    def __init__(self, chars_per_token=4):
        self.chars_per_token = chars_per_token

    def __call__(self, text, return_tensors=None):
        count = len(text) // self.chars_per_token
        return SimpleNamespace(
            input_ids=torch.arange(count, dtype=torch.long).unsqueeze(0))


def make_texts(count, size, tag='x'):
    return tuple(tag * size for _ in range(count))


# --- allocate ---------------------------------------------------------------

def test_measured_shares_cover_every_subset():
    assert set(MEASURED) == set(SUBSETS)


def test_sixteen_follows_the_measured_ratio():
    """n=16。web 2成分が 10本、残り5成分が最低1本ずつ。"""
    assert allocate(MEASURED, 16) == EXPECTED_16


def test_eight_gives_the_spare_to_the_largest():
    """成分数ぎりぎりの n=8 では、1本ずつ配った残り1本が最大成分へ行く。"""
    quota = allocate(MEASURED, 8)
    assert quota['RedPajamaCommonCrawl'] == 2
    assert sum(quota.values()) == 8
    assert all(count >= 1 for count in quota.values())


@pytest.mark.parametrize('count', [7, 8, 12, 16, 32, 64, 128])
def test_totals_and_floor_hold(count):
    quota = allocate(MEASURED, count)
    assert sum(quota.values()) == count
    assert all(value >= 1 for value in quota.values())


def test_larger_totals_approach_the_true_ratio():
    """総量を増やすと実比率に近づく。最低1本の下駄は相対的に軽くなる。"""
    share = MEASURED['RedPajamaCommonCrawl'] / sum(MEASURED.values())

    def gap(count):
        return abs(allocate(MEASURED, count)['RedPajamaCommonCrawl'] / count - share)

    assert gap(128) < gap(16) < gap(8)


def test_ties_go_to_the_first_name():
    """全成分が同率なら、余りは名前順の先頭から配る。"""
    quota = allocate({name: 1.0 for name in SUBSETS}, 8)
    assert quota[min(SUBSETS)] == 2
    assert sum(quota.values()) == 8


def test_result_does_not_depend_on_dict_order():
    reversed_shares = dict(reversed(list(MEASURED.items())))
    assert allocate(reversed_shares, 16) == allocate(MEASURED, 16)


def test_too_few_samples_is_refused():
    with pytest.raises(ValueError, match='1本ずつ配れない'):
        allocate(MEASURED, 6)


def test_non_positive_shares_are_refused():
    with pytest.raises(ValueError, match='正でない値'):
        allocate({name: 0 for name in SUBSETS}, 16)
    negative = dict(MEASURED, RedPajamaBook=-30.0)
    with pytest.raises(ValueError, match='正でない値'):
        allocate(negative, 16)


# --- 文書を集めるところ -----------------------------------------------------

def test_gather_stops_once_the_need_is_met():
    texts = make_texts(100, 1000)
    picked, size = _gather(texts, 4500, random.Random('s'))
    assert len(picked) == 5
    assert size == 5000


def test_gather_takes_everything_when_the_pool_is_short():
    texts = make_texts(3, 10)
    picked, size = _gather(texts, 10_000, random.Random('s'))
    assert len(picked) == 3
    assert size == 30


def test_gather_is_determined_by_the_stream():
    texts = make_texts(50, 100)
    first, _ = _gather(texts, 400, random.Random('0:A'))
    again, _ = _gather(texts, 400, random.Random('0:A'))
    other, _ = _gather(texts, 400, random.Random('1:A'))
    assert first == again
    assert first != other


# --- draw_component ---------------------------------------------------------

def test_draw_component_returns_the_requested_windows():
    block, component = draw_component(
        FakeTokenizer(), make_texts(400, 4096), seqlen=128, count=4,
        seed=0, name='RedPajamaC4')

    assert tuple(block.shape) == (4, 128)
    assert component.name == 'RedPajamaC4'
    assert len(component.starts) == 4
    assert component.n_tokens >= 4 * 128 * MIN_MARGIN
    assert component.metadata()['n_sequences'] == 4
    assert component.metadata()['chars_per_token'] == pytest.approx(4, abs=0.1)


def test_drawn_windows_never_overlap():
    _, component = draw_component(
        FakeTokenizer(), make_texts(400, 4096), seqlen=128, count=6,
        seed=0, name='RedPajamaCommonCrawl')

    starts = component.starts
    assert len(set(starts)) == len(starts)
    for i, left in enumerate(starts):
        for right in starts[i + 1:]:
            assert not intervals_overlap(left, right, 128)


def test_windows_are_cut_at_their_start_positions():
    """戻り値の行が、記録した開始位置そのものから切られている。"""
    tokenizer = FakeTokenizer()
    block, component = draw_component(
        tokenizer, make_texts(400, 4096), seqlen=128, count=3,
        seed=0, name='RedPajamaGithub')

    for row, start in zip(block, component.starts):
        assert row[0].item() == start
        assert row[-1].item() == start + 127


def test_draw_component_is_reproducible_and_seed_sensitive():
    texts = make_texts(400, 4096)
    first = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    again = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    other = draw_component(FakeTokenizer(), texts, 128, 4, 1, 'RedPajamaC4')[1]
    assert first == again
    assert first != other


def test_components_with_different_names_draw_differently():
    """成分ごとに独立の種。同じ文書でも名前が違えば同じ引きにならない。"""
    texts = make_texts(400, 4096)
    left = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    right = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaBook')[1]
    assert left.starts != right.starts


def test_gathered_pool_matches_the_margin():
    """集める量は窓の総面積の MARGIN 倍。密度がここで決まる。"""
    _, component = draw_component(
        FakeTokenizer(), make_texts(4000, 512), seqlen=128, count=4,
        seed=0, name='RedPajamaC4')
    assert component.n_tokens == pytest.approx(4 * 128 * MARGIN, rel=0.02)


def test_a_pool_that_is_too_small_is_refused():
    """文書を使い切っても足りないときは、黙って密度を上げずに止まる。"""
    with pytest.raises(ValueError, match='RedPajamaArXiv'):
        draw_component(FakeTokenizer(), make_texts(2, 100), seqlen=128,
                       count=2, seed=0, name='RedPajamaArXiv')


def test_a_denser_tokenizer_is_refused_rather_than_silently_crowded():
    """1トークン4文字の見積りが外れた成分も、密度が落ちれば止まる。"""
    texts = make_texts(64, 4096)
    draw_component(FakeTokenizer(4), texts, 128, 4, 0, 'RedPajamaGithub')
    with pytest.raises(ValueError, match='RedPajamaGithub'):
        draw_component(FakeTokenizer(40), texts, 128, 4, 0, 'RedPajamaGithub')


# --- 除外と札（fit / validation を分けるための2つ）---------------------------

def test_gather_without_exclusions_is_unchanged():
    """除外が空なら、除外を持たなかったころと1つも変わらない。"""
    texts = make_texts(50, 100)
    plain, _ = _gather(texts, 400, random.Random('0:A'))
    empty, _ = _gather(texts, 400, random.Random('0:A'), ())
    assert plain == empty


def test_gather_skips_excluded_documents():
    """除外した番号は出ない。シャッフルは全番号の上で先に済ませてある。"""
    texts = make_texts(50, 100)
    first, _ = _gather(texts, 400, random.Random('0:A'))
    rest, _ = _gather(texts, 400, random.Random('0:A'), first)
    assert not set(first) & set(rest)


def test_no_part_label_keeps_the_existing_stream():
    """``part=None`` は既存の全測定が通った経路そのものである。"""
    texts = make_texts(400, 4096)
    plain = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    explicit = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4',
                              part=None)[1]
    assert plain == explicit


def test_a_part_label_changes_the_draw():
    texts = make_texts(400, 4096)
    carve = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    fit = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4',
                         part='fit')[1]
    assert carve.documents != fit.documents
    assert carve.starts != fit.starts


def test_excluded_documents_stay_out_of_the_draw():
    texts = make_texts(400, 4096)
    carve = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4')[1]
    fit = draw_component(FakeTokenizer(), texts, 128, 4, 0, 'RedPajamaC4',
                         part='fit', exclude=carve.documents)[1]
    assert not set(carve.documents) & set(fit.documents)


# --- _draw_set --------------------------------------------------------------

def fake_pool():
    return {name: make_texts(400, 4096) for name in SUBSETS}


def test_the_carve_draw_survives_the_split_path():
    """``splits`` の carve は ``calibration`` と1トークンも変わらない。

    report/07 が探した配分は、この carve から作られた分割の上でしか意味を
    持たない。ここが動いたら、既存の測定とつながらなくなる。
    """
    grouped = fake_pool()
    shares = _pool(grouped)
    used = {name: set() for name in SUBSETS}
    calib = _draw_set(FakeTokenizer(), grouped, shares, 128, 8, 0,
                      'calib', None)
    carve = _draw_set(FakeTokenizer(), grouped, shares, 128, 8, 0,
                      'carve', None, used)
    assert torch.equal(calib.input_ids, carve.input_ids)


def test_later_sets_avoid_the_documents_already_used():
    grouped = fake_pool()
    shares = _pool(grouped)
    used = {name: set() for name in SUBSETS}
    carve = _draw_set(FakeTokenizer(), grouped, shares, 128, 8, 0,
                      'carve', None, used)
    fit = _draw_set(FakeTokenizer(), grouped, shares, 128, 8, 0,
                    'fit', 'fit', used)
    validation = _draw_set(FakeTokenizer(), grouped, shares, 128, 8, 0,
                           'validation', 'validation', used)

    for left, right in ((carve, fit), (carve, validation), (fit, validation)):
        for one, other in zip(left.components, right.components):
            assert not set(one.documents) & set(other.documents)


def test_every_set_keeps_the_stratified_quota():
    grouped = fake_pool()
    shares = _pool(grouped)
    used = {name: set() for name in SUBSETS}
    for count, part in ((8, None), (16, 'fit'), (16, 'validation')):
        tokens = _draw_set(FakeTokenizer(), grouped, shares, 128, count, 0,
                           'x', part, used)
        assert tokens.n_sequences == count
        counts = {row['name']: row['n_sequences']
                  for row in tokens.metadata()['components']}
        assert counts == allocate(shares, count)


# --- _documents -------------------------------------------------------------

def write_shard(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        'text': [text for text, _ in rows],
        'meta': [{'redpajama_set_name': name} for _, name in rows],
    })
    pq.write_table(table, path)
    return str(path)


def test_documents_group_by_subset(tmp_path):
    rows = [(f'{name}-{index}', name)
            for name in SUBSETS for index in range(2)]
    rows.append(('よそのコーパス', 'RedPajamaSomethingElse'))
    grouped = _documents(write_shard(tmp_path / 'shard.parquet', rows))

    assert list(grouped) == list(SUBSETS)
    assert all(len(texts) == 2 for texts in grouped.values())
    assert grouped['RedPajamaC4'] == ('RedPajamaC4-0', 'RedPajamaC4-1')


def test_a_shard_missing_a_subset_is_refused(tmp_path):
    rows = [(f'{name}-0', name) for name in SUBSETS[:-1]]
    with pytest.raises(ValueError, match=SUBSETS[-1]):
        _documents(write_shard(tmp_path / 'partial.parquet', rows))


# --- 実物 -------------------------------------------------------------------

@pytest.mark.skipif(not INTEGRATION,
                    reason='CMOE_SLIMPAJAMA_INTEGRATION=1 で走る')
def test_draws_a_stratified_set():
    """実際に引く。内訳が metadata に残り、同じ seed で同じ列が出る。"""
    from cmoe.data.slimpajama import calibration

    model = 'meta-llama/Llama-2-7b-hf'
    tokens = calibration(model, 2048, 16, 0)
    assert tuple(tokens.input_ids.shape) == (16, 2048)

    meta = tokens.metadata()
    assert meta['starts'] == []
    counts = {row['name']: row['n_sequences'] for row in meta['components']}
    assert counts == EXPECTED_16

    # プロセスをまたいで同じ列が出ることの担保。ここが変わったら、シャードか
    # 引き方のどちらかが変わっている
    assert meta['token_hash'].startswith('37738f7c8d47')

    other = calibration(model, 2048, 16, 1)
    assert other.metadata()['token_hash'] != meta['token_hash']


@pytest.mark.skipif(not INTEGRATION,
                    reason='CMOE_SLIMPAJAMA_INTEGRATION=1 で走る')
def test_splits_keep_the_carve_and_separate_fit_and_validation():
    """実物で、carve が calibration と一致し、3本が文書レベルで分かれる。"""
    from cmoe.data.slimpajama import calibration, splits

    model = 'meta-llama/Llama-2-7b-hf'
    parts = splits(model, 2048, 0, carve_count=16, fit_count=64,
                   validation_count=64)

    # report/07 の測定とつながる唯一の担保
    carve_hash = parts.carve.metadata()['token_hash']
    assert carve_hash.startswith('37738f7c8d47')
    assert carve_hash == calibration(model, 2048, 16, 0).metadata()['token_hash']

    assert tuple(parts.fit.input_ids.shape) == (64, 2048)
    assert tuple(parts.validation.input_ids.shape) == (64, 2048)

    for left, right in ((parts.carve, parts.fit),
                        (parts.carve, parts.validation),
                        (parts.fit, parts.validation)):
        for one, other in zip(left.components, right.components):
            assert one.name == other.name
            assert not set(one.documents) & set(other.documents)
