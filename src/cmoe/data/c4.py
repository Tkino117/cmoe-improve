"""C4（c4-new 相当）。

評価側は既存の測定と一致させる: CMoE-ref の ``datautils.get_c4_new`` は
validation の先頭 1100 文書を空白で連結し、先頭 256*seqlen トークンだけを使う。
seed に依存しない。

キャリブレーション側は、同関数の引き方（文書を1本ずつ引き、seqlen 以上の
長さがあるものから窓を切る）をそのまま使う。**この経路で carve した測定は
result_logs/ に存在しない** ので、既存の数字と突き合わせる対象はない。
"""

import random

import torch

from cmoe.data.base import TokenSet, load_tokenizer

TRAIN_FILE = 'en/c4-train.00000-of-01024.json.gz'
VALIDATION_FILE = 'en/c4-validation.00000-of-00008.json.gz'
VALIDATION_DOCS = 1100
VALIDATION_CHUNKS = 256


def _train():
    from datasets import load_dataset

    return load_dataset('allenai/c4', data_files={'train': TRAIN_FILE},
                        split='train')


def _validation():
    from datasets import load_dataset

    return load_dataset('allenai/c4', data_files={'validation': VALIDATION_FILE},
                        split='validation')


def calibration(model, seqlen, n_samples, seed, name='c4'):
    tokenizer = load_tokenizer(model)
    train = _train()
    rng = random.Random(seed)
    rows = []
    starts = []
    for _ in range(n_samples):
        while True:
            index = rng.randint(0, len(train) - 1)
            encoded = tokenizer(train[index]['text'], return_tensors='pt').input_ids
            if encoded.shape[1] >= seqlen:
                break
        start = rng.randint(0, encoded.shape[1] - seqlen - 1)
        rows.append(encoded[:, start:start + seqlen])
        starts.append(start)
    return TokenSet(f'{name}-train-calib', torch.cat(rows, dim=0), tuple(starts))


def evaluation(model, seqlen, name='c4-new'):
    tokenizer = load_tokenizer(model)
    validation = _validation()
    text = ' '.join(validation[:VALIDATION_DOCS]['text'])
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :VALIDATION_CHUNKS * seqlen]
    return TokenSet(f'{name}-validation', ids)
