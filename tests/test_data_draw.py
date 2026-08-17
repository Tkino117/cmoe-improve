"""キャリブレーションの引き方が、既存の測定と同じであることの確認。

トークナイザもモデルも要らない。開始位置の列だけを、CMoE-ref の
``datautils.get_wikitext2`` が使う書き方（``random.seed`` + ``random.randint``）
と突き合わせる。ここがずれると、以降の数字はすべて別の系のものになる。
"""

import random

from cmoe.data.wikitext2 import draw_starts


def legacy_starts(total_tokens, seqlen, count, seed):
    """CMoE-ref datautils.get_wikitext2 の引き方をそのまま書いたもの。"""
    random.seed(seed)
    return tuple(random.randint(0, total_tokens - seqlen - 1) for _ in range(count))


def test_draw_matches_legacy():
    total, seqlen = 2_500_000, 2048
    for seed in (0, 1, 2):
        assert draw_starts(total, seqlen, 8, seed) == legacy_starts(total, seqlen, 8, seed)


def test_larger_n_extends_the_same_draw():
    """n を増やしても最初の8本は変わらない。n=8 の測定と比較できる条件。"""
    total, seqlen = 2_500_000, 2048
    assert draw_starts(total, seqlen, 64, 0)[:8] == draw_starts(total, seqlen, 8, 0)
