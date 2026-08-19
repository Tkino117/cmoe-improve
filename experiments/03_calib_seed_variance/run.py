"""03 SA配分キャリブレーションの適正量。設定は同じディレクトリの README。

探索用 carve の seed（docs/02 の①）だけを振って配分を探し、出た配分を
変換用 carve seed 0 固定で評価して、PPL のばらつきを見る。

  uv run python experiments/03_calib_seed_variance/run.py --oracle local_error --seeds 0,1,2,3,4

探索は seed ごとに1回ずつ、評価は全配分をまとめて1回の run で回す（変換は
配分ごとに1度で済み、モデルの読み込みも1回になる）。途中で落ちても、済んだ
探索の出力ディレクトリはそのまま残るので、同じコマンドで続きから走る。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 振らないもの。README の「固定するもの」と同じ
WIDTH = 2
SEARCH = 'beam'
ROUTER = 'cmoe'
BUILD_SEED = 0            # ② 変換用 carve。固定
DATASETS = 'wikitext2,c4-new'
BASELINE = 'uniform3'


def run_cli(argv):
    command = ['uv', 'run', 'cmoe', *argv]
    print(f'$ {" ".join(command)}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def search_one(oracle, seed, out_dir):
    """seed ごとの探索。済んでいれば飛ばす。"""
    payload = out_dir / 'search.json'
    if payload.exists():
        print(f'  seed {seed}: 済み（{payload} を使う）', flush=True)
    else:
        run_cli(['search', '--search', SEARCH, '--width', str(WIDTH),
                 '--oracle', oracle, '--seed', str(seed),
                 '--out', str(out_dir)])
    with payload.open() as handle:
        record = json.load(handle)
    return record['allocation']['values'], record['seconds']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', required=True,
                        help='採点オラクル（local_error / suffix_kl など）')
    parser.add_argument('--seeds', required=True,
                        help='探索用 carve の seed。カンマ区切り')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    if len(seeds) < 2:
        raise SystemExit('ばらつきを見るので seed は2本以上')

    out = Path(args.out or
               ROOT / f'result_logs/calib_seed_variance_{args.oracle}_w{WIDTH}')
    out.mkdir(parents=True, exist_ok=True)

    allocations = {}
    search_seconds = 0.0
    for seed in seeds:
        values, seconds = search_one(args.oracle, seed, out / f'search_seed{seed}')
        allocations[seed] = ','.join(str(value) for value in values)
        search_seconds += seconds

    print('\n== seed ごとの配分 ==', flush=True)
    for seed, vector in allocations.items():
        print(f'  seed {seed}: {vector}', flush=True)

    # 同じベクトルが2度出たら --alloc を重ねない（同じ spec は同じ構成として
    # 集計されるので、重ねると行が衝突する）
    unique = []
    for vector in allocations.values():
        if vector not in unique:
            unique.append(vector)
    if len(unique) < len(allocations):
        print(f'  {len(allocations)} 本中 {len(unique)} 本が相異なる', flush=True)

    # 評価が済んでいれば飛ばす（cmoe run は終わった出力先への上書きを拒む）
    if (out / 'eval' / 'summary.json').exists():
        print(f'  評価: 済み（{out / "eval" / "summary.json"} を使う）', flush=True)
    else:
        argv = ['run']
        for vector in [BASELINE, *unique]:
            argv += ['--alloc', vector]
        argv += ['--router', ROUTER, '--seeds', str(BUILD_SEED),
                 '--datasets', DATASETS, '--out', str(out / 'eval')]
        run_cli(argv)

    summarize(out, allocations, unique, search_seconds)


def summarize(out, allocations, unique, search_seconds):
    """seed ごとの PPL を並べ、標準偏差を出す。"""
    with (out / 'eval' / 'summary.json').open() as handle:
        payload = json.load(handle)
    # summary は行の並びではなく {'configurations': [...]} である
    rows = payload['summary']['configurations']
    eval_seconds = sum(record['seconds'] for record in payload['runs'])
    ppl = {row['allocation']: {name: entry['mean_ppl']
                               for name, entry in row['datasets'].items()}
           for row in rows}

    lines = ['# 03 キャリブレーション量: seed 変化と分散', '',
             f'探索 {SEARCH} 幅{WIDTH} / 変換用 carve seed {BUILD_SEED} 固定 / '
             f'評価 {DATASETS}', '']
    datasets = [name for name in DATASETS.split(',')]
    header = ' | '.join(['seed', '平均 x', *datasets])
    lines += [f'| {header} |', '|' + '---|' * (len(datasets) + 2)]

    values = {name: [] for name in datasets}
    for seed, vector in allocations.items():
        mean_x = sum(int(x) for x in vector.split(',')) / len(vector.split(','))
        cells = []
        for name in datasets:
            score = ppl[vector][name]
            values[name].append(score)
            cells.append(f'{score:.6f}')
        lines.append(f'| {seed} | {mean_x:.2f} | ' + ' | '.join(cells) + ' |')

    base = ppl[BASELINE]
    lines.append(f'| {BASELINE} | — | '
                 + ' | '.join(f'{base[name]:.6f}' for name in datasets) + ' |')
    lines += ['', f'相異なる配分 {len(unique)}/{len(allocations)} 本', '']

    for name in datasets:
        series = values[name]
        mean = sum(series) / len(series)
        variance = sum((value - mean) ** 2 for value in series) / (len(series) - 1)
        deviation = variance ** 0.5
        lines.append(f'- **{name}**: 平均 {mean:.6f} / 標準偏差 {deviation:.6f} / '
                     f'幅 {min(series):.6f}〜{max(series):.6f} / '
                     f'{BASELINE} との差 {mean - base[name]:+.6f}')

    lines += ['', f'探索 {search_seconds / 60:.1f} 分 / '
                  f'評価 {eval_seconds / 60:.1f} 分', '']
    report = '\n'.join(lines)
    (out / 'variance.md').write_text(report, encoding='utf-8')
    print('\n' + report, flush=True)
    print(f'書き出し: {out / "variance.md"}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
