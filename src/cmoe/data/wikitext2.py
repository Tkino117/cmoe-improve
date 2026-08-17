"""WikiText-2。キャリブレーションの引き方は既存の測定と一致させてある。

引き方の同一性が要点である。CMoE-ref の ``datautils.get_wikitext2`` は
``random.seed(seed)`` の上で ``random.randint(0, N - seqlen - 1)`` を nsamples 回
引く。ここでは ``random.Random(seed)`` を使うが、同じ Mersenne Twister を同じ
種で初期化して同じメソッドを呼ぶので、出てくる開始位置の列は一致する。

したがって seed 0・n=8 で引ける8本は、result_logs/ にある全測定が carve に
使ったのと同じ8本である。
"""

import random

from cmoe.data.base import Splits, TokenSet, load_tokenizer


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


def _validation_text():
    from datasets import load_dataset

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    return '\n\n'.join(data['text'])


def intervals_overlap(left, right, length):
    """同じ長さの半開区間が重なるか。"""
    return left < right + length and right < left + length


def disjoint_starts(total_tokens, seqlen, carve_count, fit_count, seed,
                    max_draws=1_000_000):
    """carve の引きをそのまま残し、続けて重ならない fit を引く。

    最初の carve_count 本は ``draw_starts`` と同一である（同じ RNG ストリームの
    先頭）。fit はその続きを引き、carve とも先に採った fit とも重なるものを
    捨てて集める。
    """
    if total_tokens <= seqlen:
        raise ValueError(f'{total_tokens} トークンでは seqlen={seqlen} を切り出せない')
    if carve_count < 1 or fit_count < 1:
        raise ValueError('carve_count と fit_count は正のはず')
    if max_draws < fit_count:
        raise ValueError('max_draws は fit_count 以上のはず')

    rng = random.Random(seed)
    upper = total_tokens - seqlen - 1
    carve = [rng.randint(0, upper) for _ in range(carve_count)]
    occupied = list(carve)
    fit = []
    draws = 0
    while len(fit) < fit_count and draws < max_draws:
        start = rng.randint(0, upper)
        draws += 1
        if any(intervals_overlap(start, previous, seqlen) for previous in occupied):
            continue
        fit.append(start)
        occupied.append(start)
    if len(fit) != fit_count:
        raise ValueError(
            f'{draws} 回引いて重ならない fit が {len(fit)}/{fit_count} 本しか置けなかった')
    return tuple(carve), tuple(fit)


def contiguous_starts(total_tokens, seqlen, count):
    if count < 1:
        raise ValueError('validation の本数は正のはず')
    required = count * seqlen
    if total_tokens < required:
        raise ValueError(
            f'validation は {total_tokens} トークン、{required} 必要')
    return tuple(index * seqlen for index in range(count))


def splits(model, seqlen, seed, carve_count=8, fit_count=64,
           validation_count=64, name='wikitext2'):
    """carve / fit / validation の3本。fit は carve と重ならない。

    validation は訓練 split ですらない（wikitext2 の validation split の先頭から
    連続で切る）ので、seed に依存しない。
    """
    tokenizer = load_tokenizer(model)
    train_ids = tokenizer(_train_text(), return_tensors='pt').input_ids
    validation_ids = tokenizer(_validation_text(), return_tensors='pt').input_ids

    carve_starts, fit_starts = disjoint_starts(
        train_ids.shape[1], seqlen, carve_count, fit_count, seed)
    validation_starts = contiguous_starts(
        validation_ids.shape[1], seqlen, validation_count)
    return Splits(
        carve=TokenSet(f'{name}-train-carve',
                       take(train_ids, carve_starts, seqlen), carve_starts),
        fit=TokenSet(f'{name}-train-fit',
                     take(train_ids, fit_starts, seqlen), fit_starts),
        validation=TokenSet(f'{name}-validation',
                            take(validation_ids, validation_starts, seqlen),
                            validation_starts),
    )
