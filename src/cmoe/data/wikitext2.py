"""WikiText-2。キャリブレーションの引き方は既存の測定と一致させてある。

引き方の同一性が要点である。CMoE-ref の ``datautils.get_wikitext2`` は
``random.seed(seed)`` の上で ``random.randint(0, N - seqlen - 1)`` を nsamples 回
引く。ここでは ``random.Random(seed)`` を使うが、同じ Mersenne Twister を同じ
種で初期化して同じメソッドを呼ぶので、出てくる開始位置の列は一致する。

したがって seed 0・n=8 で引ける8本は、result_logs/ にある全測定が carve に
使ったのと同じ8本である。
"""

import random

from cmoe.data.base import TokenSet, load_tokenizer


def _train_text():
    from datasets import load_dataset

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    return '\n\n'.join(data['text'])


def _test_text():
    from datasets import load_dataset

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return '\n\n'.join(data['text'])


def draw_starts(total_tokens, seqlen, count, seed):
    """既存の引き方をそのまま再現する開始位置の列。"""
    rng = random.Random(seed)
    upper = total_tokens - seqlen - 1
    if upper < 0:
        raise ValueError(f'{total_tokens} トークンでは seqlen={seqlen} を切り出せない')
    return tuple(rng.randint(0, upper) for _ in range(count))


def take(input_ids, starts, seqlen):
    import torch

    return torch.cat([input_ids[:, start:start + seqlen] for start in starts], dim=0)


def calibration(model, seqlen, n_samples, seed, name='wikitext2'):
    tokenizer = load_tokenizer(model)
    ids = tokenizer(_train_text(), return_tensors='pt').input_ids
    starts = draw_starts(ids.shape[1], seqlen, n_samples, seed)
    return TokenSet(f'{name}-train-calib', take(ids, starts, seqlen), starts)


def evaluation(model, seqlen, name='wikitext2'):
    """テスト split 全体を1本に連結したもの。seed には依存しない。"""
    tokenizer = load_tokenizer(model)
    ids = tokenizer(_test_text(), return_tensors='pt').input_ids
    return TokenSet(f'{name}-test', ids)
