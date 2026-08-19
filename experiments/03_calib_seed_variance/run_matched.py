"""03b 探索用 carve と変換用 carve の seed を揃えたときのばらつき。

``run.py`` は変換用 carve（docs/02 の②）を seed 0 に固定して探索用（①）だけを
振った。実運用ではキャリブレーションは1セットしか無く、①と②は同じ8本になる。
ここでは①=②=s に揃えて測り直す。

探索はやり直さない。``run.py`` が残した search.json から配分ベクトルを読むだけで、
探索側の出力には一切書き込まない。

  uv run python experiments/03_calib_seed_variance/run_matched.py \
      --oracle suffix_kl --seeds 0,1,2,3,4

対照 uniform3 も seed ごとに組み直す（②が振れるので対照側も動く）。
seed ごとに run を1回ずつ呼ぶので、各 seed 内で対応の取れた比較になる。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WIDTH = 2
ROUTER = 'cmoe'
DATASETS = 'wikitext2,c4-new'
BASELINE = 'uniform3'


def run_cli(argv):
    command = ['uv', 'run', 'cmoe', *argv]
    print(f'$ {" ".join(command)}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--source', default=None,
                        help='配分を読む探索の出力先。既定は run.py の出力')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    if len(seeds) < 2:
        raise SystemExit('ばらつきを見るので seed は2本以上')

    source = Path(args.source or
                  ROOT / f'result_logs/calib_seed_variance_{args.oracle}_w{WIDTH}')
    out = Path(args.out or f'{source}_matched')
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == source.resolve():
        raise SystemExit('出力先が探索の出力先と同じ。上書きになる')

    allocations = {}
    for seed in seeds:
        payload = source / f'search_seed{seed}' / 'search.json'
        if not payload.exists():
            raise SystemExit(f'{payload} が無い。先に run.py を走らせる')
        with payload.open() as handle:
            values = json.load(handle)['allocation']['values']
        allocations[seed] = ','.join(str(value) for value in values)

    print('== 読んだ配分（探索はやり直さない） ==', flush=True)
    for seed, vector in allocations.items():
        print(f'  seed {seed}: {vector}', flush=True)

    for seed in seeds:
        eval_dir = out / f'eval_seed{seed}'
        if (eval_dir / 'summary.json').exists():
            print(f'  seed {seed}: 評価済み', flush=True)
            continue
        run_cli(['run', '--alloc', BASELINE, '--alloc', allocations[seed],
                 '--router', ROUTER, '--seeds', str(seed),
                 '--datasets', DATASETS, '--out', str(eval_dir)])

    summarize(out, allocations, seeds)


def deviation(series):
    mean = sum(series) / len(series)
    variance = sum((value - mean) ** 2 for value in series) / (len(series) - 1)
    return mean, variance ** 0.5


def summarize(out, allocations, seeds):
    datasets = DATASETS.split(',')
    searched = {name: [] for name in datasets}
    control = {name: [] for name in datasets}
    paired = {name: [] for name in datasets}
    seconds = 0.0

    lines = ['# 03b 探索用 carve と変換用 carve を揃えたときのばらつき', '',
             f'①=②=seed / 探索 beam 幅{WIDTH} / 評価 {DATASETS}', '']
    header = ['seed', '平均 x']
    for name in datasets:
        header += [f'{name} 対照', f'{name} 探索', f'{name} NLL 差']
    lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]

    for seed in seeds:
        with (out / f'eval_seed{seed}' / 'summary.json').open() as handle:
            payload = json.load(handle)
        seconds += sum(record['seconds'] for record in payload['runs'])
        rows = {row['allocation']: row for row in payload['summary']['configurations']}
        vector = allocations[seed]
        mean_x = sum(int(x) for x in vector.split(',')) / len(vector.split(','))
        cells = [str(seed), f'{mean_x:.2f}']
        for name in datasets:
            base = rows[BASELINE]['datasets'][name]['mean_ppl']
            score = rows[vector]['datasets'][name]['mean_ppl']
            stats = rows[vector]['datasets'][name]['paired_vs_baseline']
            control[name].append(base)
            searched[name].append(score)
            paired[name].append(stats)
            cells += [f'{base:.6f}', f'{score:.6f}',
                      f'{stats["mean_nll_difference"]:+.6f} '
                      f'[{stats["lower"]:+.6f}, {stats["upper"]:+.6f}]']
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.append('')
    for name in datasets:
        mean, spread = deviation(searched[name])
        base_mean, base_spread = deviation(control[name])
        differences = [stats['mean_nll_difference'] for stats in paired[name]]
        diff_mean, diff_spread = deviation(differences)
        negative = sum(1 for value in differences if value < 0)
        excludes = sum(1 for stats in paired[name]
                       if stats['upper'] < 0 or stats['lower'] > 0)
        lines += [
            f'## {name}',
            '',
            f'- 探索配分: 平均 {mean:.6f} / **標準偏差 {spread:.6f}** / '
            f'幅 {min(searched[name]):.6f}〜{max(searched[name]):.6f}',
            f'- 対照 {BASELINE}: 平均 {base_mean:.6f} / **標準偏差 {base_spread:.6f}** / '
            f'幅 {min(control[name]):.6f}〜{max(control[name]):.6f}',
            f'- NLL 差: 平均 {diff_mean:+.6f} / 標準偏差 {diff_spread:.6f} / '
            f'改善 {negative}/{len(differences)} seed / '
            f'区間が0を含まない {excludes}/{len(differences)} seed',
            '',
        ]
    lines.append(f'評価 {seconds / 60:.1f} 分')

    report = '\n'.join(lines)
    (out / 'variance.md').write_text(report, encoding='utf-8')
    print('\n' + report, flush=True)
    print(f'\n書き出し: {out / "variance.md"}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
