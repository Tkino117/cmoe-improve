"""評価 seed で引き直す評価セットの確認。

トークナイザもモデルも要らない。seed 0 が既存の測定と同じ切り方になることと、
seed どうしが重ならないことだけを見る。
"""

import pytest

from cmoe.data.c4 import VALIDATION_DOCS, document_range
from cmoe.data.registry import load_evaluation
from cmoe.data.wikitext2 import evaluation_offset


def test_seed_zero_matches_existing():
    assert evaluation_offset(0, 2048) == 0
    assert document_range(0) == (0, VALIDATION_DOCS)


def test_wikitext2_offsets_are_distinct_within_a_chunk():
    offsets = [evaluation_offset(seed, 2048) for seed in range(11)]
    assert len(set(offsets)) == 11
    assert all(0 <= offset < 2048 for offset in offsets)


def test_c4_documents_do_not_overlap():
    ranges = [document_range(seed) for seed in range(11)]
    for (_, stop), (start, _) in zip(ranges, ranges[1:]):
        assert stop == start


@pytest.mark.parametrize('name', ['wikitext2@x', 'unknown@1'])
def test_malformed_seeded_name_is_rejected(name):
    with pytest.raises(ValueError):
        load_evaluation(name, 'meta-llama/Llama-2-7b-hf', 2048)
