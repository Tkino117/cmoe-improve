"""25 の結果を、seed をまたいだ表にまとめる。GPU は要らない。

**途中でも読める。** 揃っている seed だけで平均し、各行に n を出す。走らせながら
現状を見るのに使える（20ジョブが終わるまで待たなくてよい）。

数値は各実行の ``summary.json`` に集計済みのものを読むだけで、生の尤度
（``bench/run*.json``）は開かない。指標を足したいときだけそちらを見る
（``cmoe.eval.bench.load_samples``）。

**行の名前は seed をまたいで対応させる。** 探索が出す配分は seed ごとに違うので、
配分ベクトルでは束ねられない。``slimpajama*_w<W>_n16_seed<N>/search.json`` を
引いて「その seed の幅W が出した配分」を突き合わせ、`探索 幅W` という行に
まとめる。一様は名前がそのまま行になる。

  uv run python experiments/25_model_seeds/summarize.py
  uv run python experiments/25_model_seeds/summarize.py --models mistral-7b --nactives 6
  uv run python experiments/25_model_seeds/summarize.py --metric acc,gold_nll,ppl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CALIB = 'slimpajama'
NSAMPLES = 16
NEXPERTS = 8
MODELS = ('llama2-7b', 'mistral-7b')
UNTAGGED_MODEL = 'llama2-7b'
NACTIVES = (6, 4)
SEEDS = (0, 1, 2, 3, 4)
WIDTHS = (2, 3, 4)
TASKS = ('piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag')
SHORT = {'piqa': 'PIQA', 'winogrande': 'Wino', 'arc_easy': 'ARC-e',
         'arc_challenge': 'ARC-c', 'hellaswag': 'HSwag', 'mmlu': 'MMLU'}
DATASETS = ('wikitext2', 'c4-new')
ROUTER = 'cmoe'


def log(message=''):
    print(message)


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


# -- 出力先の名前（run.py と同じ規則） ------------------------------------

def model_tag(model):
    return '' if model == UNTAGGED_MODEL else f'_{model}'


def sparsity_tag(n_active):
    return '' if n_active == 6 else f'_a{n_active}'


def stem(model, n_active):
    return f'{CALIB}{model_tag(model)}{sparsity_tag(n_active)}'


def bench_dir(root, model, n_active, seed, mmlu=False):
    prefix = 'bench_mmlu' if mmlu else 'bench'
    return root / f'{prefix}_{stem(model, n_active)}_seed{seed}'


def search_dir(root, model, n_active, seed, width):
    return root / f'{stem(model, n_active)}_w{width}_n{NSAMPLES}_seed{seed}'


# -- 行の名前 -------------------------------------------------------------

def labels_for(root, model, n_active, seed):
    """配分ベクトル（カンマ区切り）→ その配分が当たる行の名前（複数）。

    **幅が違っても同じ配分に着くことがある**（report/15 の seed 2 は 幅2 と 幅3 が
    同値だった）。そのとき1つの行に束ねると、束ねた側の n が減って seed 平均が
    別物になる。**両方の行に同じ値を入れる**のが正しい — 「幅3 が出した配分」は
    確かにその配分だからである。一様は ``summary.json`` 側の名前を使うので
    ここには入らない。
    """
    found = {}
    for width in WIDTHS:
        payload = search_dir(root, model, n_active, seed, width) / 'search.json'
        if not payload.exists():
            continue
        record = load(payload)
        if 'allocation' not in record:
            continue
        spec = ','.join(str(value) for value in record['allocation']['values'])
        found.setdefault(spec, []).append(f'探索 幅{width}')
    return found


def row_order(n_active, rows=None):
    """表の縦の並び。x の小さい順の一様、そのあと探索を幅の順に。

    ``rows`` を渡すと、並びに無い名前を末尾に足す。**黙って落とさない**ため。
    """
    order = ([f'uniform{x}' for x in range(n_active + 1)]
             + [f'探索 幅{width}' for width in WIDTHS])
    if rows is not None:
        order += [name for name in rows if name not in order]
    return order


# -- 収集 -----------------------------------------------------------------

def collect(root, model, n_active, seeds):
    """(行の名前) → {指標: [seed ごとの値]}。無い seed は飛ばす。"""
    rows = {}
    used = []
    for seed in seeds:
        payload = bench_dir(root, model, n_active, seed) / 'summary.json'
        if not payload.exists():
            continue
        record = load(payload)
        if 'bench_summary' not in record or record.get('failures'):
            continue
        used.append(seed)
        names = labels_for(root, model, n_active, seed)
        # MMLU は別の実行。あれば同じ行に足す
        mmlu_payload = bench_dir(root, model, n_active, seed, mmlu=True) / 'summary.json'
        mmlu = load(mmlu_payload) if mmlu_payload.exists() else None

        for index, run in enumerate(record['runs']):
            allocation = run['allocation']
            spec = ','.join(str(value) for value in allocation['values'])
            for label in names.get(spec, [allocation['name']]):
                entry = rows.setdefault(label, {})
                entry.setdefault('mean_x', []).append(allocation['mean_x'])
                for task, values in run['bench'][ROUTER]['tasks'].items():
                    for metric in ('acc', 'gold_nll'):
                        entry.setdefault(f'{metric}/{task}', []).append(
                            values[metric])
                for dataset in DATASETS:
                    got = run['ppl'][ROUTER].get(dataset) if 'ppl' in run else None
                    if got:
                        entry.setdefault(f'ppl/{dataset}', []).append(got['ppl'])
                if mmlu is not None and index < len(mmlu['runs']):
                    got = mmlu['runs'][index]
                    # 同じ配分を測っているか（順番は run.py が揃えている）
                    if got['allocation']['values'] == allocation['values']:
                        for metric in ('acc', 'gold_nll'):
                            entry.setdefault(f'{metric}/mmlu', []).append(
                                got['bench'][ROUTER]['tasks']['mmlu'][metric])
    return rows, used


def cell(values, digits=4):
    if not values:
        return '—'
    if len(values) == 1:
        return f'{values[0]:.{digits}f}'
    return (f'{statistics.mean(values):.{digits}f}'
            f' ± {statistics.stdev(values):.{digits}f}')


def best(rows, order, key, higher):
    """列ごとの最良の行。値が1つも無い列は None。"""
    candidates = [(name, statistics.mean(rows[name][key]))
                  for name in order
                  if name in rows and rows[name].get(key)]
    if not candidates:
        return None
    return (max if higher else min)(candidates, key=lambda pair: pair[1])[0]


def table(rows, order, keys, headers, higher, digits=4):
    log('| 配分 | n | ' + ' | '.join(headers) + ' |')
    log('|---|--:|' + '--:|' * len(keys))
    marks = {key: best(rows, order, key, higher) for key in keys}
    for name in order:
        if name not in rows:
            continue
        counts = {len(values) for key in keys
                  for values in [rows[name].get(key, [])] if values}
        n = max(counts) if counts else 0
        cells = []
        for key in keys:
            text = cell(rows[name].get(key, []), digits)
            if marks.get(key) == name and text != '—':
                text = f'**{text}**'
            cells.append(text)
        log(f'| {name} | {n} | ' + ' | '.join(cells) + ' |')


def report(root, model, n_active, seeds, metrics):
    rows, used = collect(root, model, n_active, seeds)
    sparsity = 100 * (NEXPERTS - n_active) // NEXPERTS
    log()
    log(f'## {model} / N={NEXPERTS} A={n_active}（スパース率 {sparsity}%）')
    if not rows:
        log()
        log('まだ結果が無い。')
        return
    log()
    log(f'seed {", ".join(str(seed) for seed in used)}（{len(used)} 本）。'
        'セルは平均 ± 標準偏差、n は揃っている seed の本数。')
    order = row_order(n_active, rows)
    tasks = [task for task in (*TASKS, 'mmlu')
             if any(f'acc/{task}' in rows[name] for name in rows)]

    if 'acc' in metrics:
        log()
        log('**`acc`**（大きいほど良い）')
        table(rows, order, [f'acc/{task}' for task in tasks],
              [SHORT[task] for task in tasks], higher=True)
    if 'gold_nll' in metrics:
        log()
        log('**`gold_nll`**（小さいほど良い）'
            ' — タスクをまたいで比べないこと（続きの長さで尺度が変わる）')
        table(rows, order, [f'gold_nll/{task}' for task in tasks],
              [SHORT[task] for task in tasks], higher=False)
    if 'ppl' in metrics:
        keys = [f'ppl/{dataset}' for dataset in DATASETS]
        if any(key in rows[name] for name in rows for key in keys):
            log()
            log('**PPL**（小さいほど良い）— モデルをまたいで比べないこと')
            table(rows, order, keys, ['WikiText-2', 'C4-new'], higher=False)
    if 'mean_x' in metrics:
        log()
        log('**平均 x**（層あたりの shared expert 数。予算は A）')
        table(rows, order, ['mean_x'], ['平均 x'], higher=False, digits=3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--nactives', default=','.join(str(a) for a in NACTIVES))
    parser.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    parser.add_argument('--metric', default='acc,gold_nll,ppl,mean_x')
    parser.add_argument('--out', default=None, help='一次データの置き場所')
    args = parser.parse_args(argv)

    root = Path(args.out).resolve() if args.out else ROOT / 'result_logs'
    models = [f.strip() for f in args.models.split(',') if f.strip()]
    nactives = [int(f) for f in args.nactives.split(',') if f.strip()]
    seeds = [int(f) for f in args.seeds.split(',') if f.strip()]
    metrics = {f.strip() for f in args.metric.split(',') if f.strip()}

    log(f'# 25 の途中集計（{root}）')
    for model in models:
        for n_active in nactives:
            report(root, model, n_active, seeds, metrics)
    return 0


if __name__ == '__main__':
    sys.exit(main())
