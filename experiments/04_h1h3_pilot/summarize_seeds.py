"""04 の複数 seed をまとめる。GPU は使わない。

各 seed の ``pilot.json`` を読み、構成ごとに seed 間の平均と標準偏差を出す。

  uv run python experiments/04_h1h3_pilot/summarize_seeds.py --seeds 0,1,2

配分は seed ごとに変わるので、突き合わせる鍵はベクトルではなく「同じ手続きで
作った配分」である。一様は名前で、探索の結果は**幅**で引く（``pilot.json`` の
``searches`` が幅 → ベクトルを持つ）。幅2と幅3が同じ配分に着いた seed でも、
行が消えずに両方の幅へ配られる。校正が変わればモデルの組み方も変わるので、
鍵は（校正, ラベル）である。

生の PPL の標準偏差には、その seed が引いた8本の当たり外れが全構成に共通で
乗る。差を見たいときはそれが打ち消し合うので、同じ seed の中で引き算してから
seed 間のばらつきを取る表も出す。
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [('wikitext2', 'wikitext2'), ('wikitext2', 'c4-new'),
           ('c4', 'wikitext2'), ('c4', 'c4-new')]
UNIFORMS = [f'uniform{x}' for x in range(7)]
SEARCHED = [f'beam 幅{width}' for width in (2, 3, 4)]
ORDER = UNIFORMS + SEARCHED
BASELINE = 'uniform3'
STAGES = {'h1', 'h2', 'h3'}


def seed_dir(seed):
    return ROOT / 'result_logs' / ('h1h3_pilot' if seed == 0
                                   else f'h1h3_pilot_seed{seed}')


def load(seed, path):
    """1 seed ぶんの pilot.json。seed と段が揃っていなければ止める。"""
    payload = path / 'pilot.json'
    if not payload.exists():
        raise SystemExit(f'{payload} が無い')
    with payload.open() as handle:
        record = json.load(handle)
    given = record.get('plan', {}).get('seed')
    if given != seed:
        raise SystemExit(f'{payload} は seed {given} の結果で、{seed} ではない')
    missing = STAGES - set(record.get('stages', {}))
    if missing:
        raise SystemExit(f'{payload} は {sorted(missing)} を欠いている')
    return record


def collect(records):
    """（校正, ラベル, 評価）→ {seed: PPL} と、（校正, ラベル）→ {seed: 平均 x}。"""
    ppl, mean_x = {}, {}
    for seed, record in records.items():
        for stage in record['stages'].values():
            calib = stage['calib']
            # 幅 → ベクトル。同じベクトルに複数の幅が着くことがある
            widths = {}
            for width, search in stage.get('searches', {}).items():
                spec = ','.join(str(x) for x in search['allocation'])
                widths.setdefault(spec, []).append(int(width))
            rows = [*stage.get('uniform_rows', []), *stage['rows']]
            for row in rows:
                found = widths.get(row['allocation'])
                labels = ([f'beam 幅{width}' for width in found] if found
                          else [row['label']])
                for label in labels:
                    if label not in ORDER:
                        raise SystemExit(
                            f'seed {seed} に見覚えのない配分 {label!r} がある')
                    mean_x.setdefault((calib, label), {})[seed] = row['mean_x']
                    for dataset, cell in row['datasets'].items():
                        ppl.setdefault((calib, label, dataset), {})[seed] = cell['ppl']
    return ppl, mean_x


def statistic(values):
    """平均・標準偏差・本数。1本では標準偏差を出さない（0 と書かない）。"""
    if not values:
        return None
    return {'mean': statistics.fmean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values), 'min': min(values), 'max': max(values)}


def cell(entry, expected, digits=6):
    if entry is None:
        return '—'
    text = f'{entry["mean"]:.{digits}f}'
    if entry['stdev'] is not None:
        text += f' ± {entry["stdev"]:.{digits}f}'
    if entry['n'] != expected:
        text += f'（n={entry["n"]}）'
    return text


def series(ppl, calib, label, dataset, seeds):
    values = ppl.get((calib, label, dataset), {})
    return [values[seed] for seed in seeds if seed in values]


def table(lines, title, note, seeds, labels, compute):
    lines += [f'### {title}', '', note, '',
              '| 配分 | ' + ' | '.join(f'校正 {calib} → {dataset}'
                                       for calib, dataset in COLUMNS) + ' |',
              '|' + '---|' * (len(COLUMNS) + 1)]
    rows = []
    for label in labels:
        record, cells = {'label': label, 'datasets': {}}, []
        for calib, dataset in COLUMNS:
            values = compute(calib, label, dataset)
            entry = statistic(values)
            record['datasets'][f'{calib}/{dataset}'] = (
                None if entry is None else {**entry, 'per_seed': values})
            cells.append(cell(entry, len(seeds)))
        rows.append(record)
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--dirs', default=None,
                        help='seed ごとの出力先。カンマ区切りで --seeds と同順')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    paths = {seed: seed_dir(seed) for seed in seeds}
    if args.dirs:
        given = [Path(field).resolve() for field in args.dirs.split(',')]
        if len(given) != len(seeds):
            raise SystemExit('--dirs は --seeds と同じ数を並べる')
        paths = dict(zip(seeds, given))

    records = {seed: load(seed, paths[seed]) for seed in seeds}
    ppl, mean_x = collect(records)
    out = (Path(args.out).resolve() if args.out
           else ROOT / 'result_logs/h1h3_pilot_seeds')
    out.mkdir(parents=True, exist_ok=True)

    labels = [label for label in ORDER
              if any((calib, label, dataset) in ppl for calib, dataset in COLUMNS)]
    lines = [f'# 04 H1〜H3 を seed {len(seeds)}本で（{", ".join(map(str, seeds))}）',
             '', 'seed は校正に引く8本を選ぶくじで、配分を探す文章①とモデルを組む'
             '文章②に同じものを使っている。', '']
    payload = {'seeds': seeds, 'sources': {str(seed): str(path)
                                           for seed, path in paths.items()},
               'repository': {str(seed): record.get('repository')
                              for seed, record in records.items()}}

    payload['ppl'] = table(
        lines, 'PPL', 'seed 間の平均 ± 標準偏差（標本、n−1）。', seeds, labels,
        lambda calib, label, dataset: series(ppl, calib, label, dataset, seeds))

    # 同じ seed の中で引くと、その seed が引いた8本の当たり外れが打ち消える
    def against(baseline):
        def compute(calib, label, dataset):
            values = []
            for seed in seeds:
                left = ppl.get((calib, baseline, dataset), {}).get(seed)
                right = ppl.get((calib, label, dataset), {}).get(seed)
                if left is not None and right is not None:
                    values.append(right - left)
            return values
        return compute

    payload['vs_baseline'] = table(
        lines, f'{BASELINE} との差（同じ seed の中で引いてから、seed 間で平均）',
        f'負なら {BASELINE} より良い。seed 効果が打ち消えるので、上の表の標準偏差'
        'よりも小さくなる。', seeds, [l for l in labels if l != BASELINE],
        against(BASELINE))

    # 各 seed で最も良かった一様配分（seed ごとに別の x でありうる）
    best = {}
    for calib, dataset in COLUMNS:
        for seed in seeds:
            candidates = [(ppl[(calib, name, dataset)][seed], name)
                          for name in UNIFORMS
                          if seed in ppl.get((calib, name, dataset), {})]
            if candidates:
                best[(calib, dataset, seed)] = min(candidates)

    def against_best(calib, label, dataset):
        values = []
        for seed in seeds:
            reference = best.get((calib, dataset, seed))
            right = ppl.get((calib, label, dataset), {}).get(seed)
            if reference and right is not None:
                values.append(right - reference[0])
        return values

    payload['vs_best_uniform'] = table(
        lines, 'その seed で最良の一様配分との差',
        '負なら、その seed のどの一様配分よりも良い。', seeds,
        [label for label in SEARCHED if label in labels], against_best)
    lines += ['最良だった一様配分:', '']
    for calib, dataset in COLUMNS:
        names = [f'seed {seed} {best[(calib, dataset, seed)][1]}'
                 for seed in seeds if (calib, dataset, seed) in best]
        lines.append(f'- 校正 {calib} / 評価 {dataset}: ' + ' / '.join(names))
    lines.append('')
    payload['best_uniform'] = {f'{calib}/{dataset}':
                               {str(seed): best[(calib, dataset, seed)][1]
                                for seed in seeds if (calib, dataset, seed) in best}
                               for calib, dataset in COLUMNS}

    lines += ['### 平均 x', '',
              '| 配分 | ' + ' | '.join(f'校正 {calib}' for calib in ('wikitext2', 'c4'))
              + ' |', '|---|---|---|']
    payload['mean_x'] = {}
    for label in labels:
        cells = []
        for calib in ('wikitext2', 'c4'):
            values = [mean_x[(calib, label)][seed] for seed in seeds
                      if seed in mean_x.get((calib, label), {})]
            entry = statistic(values)
            payload['mean_x'][f'{calib}/{label}'] = entry
            cells.append('—' if entry is None else cell(entry, len(seeds), 2))
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines += ['### seed ごとの PPL', '']
    for calib, dataset in COLUMNS:
        lines += [f'**校正 {calib} / 評価 {dataset}**', '',
                  '| 配分 | ' + ' | '.join(f'seed {seed}' for seed in seeds)
                  + ' | 幅 |', '|' + '---|' * (len(seeds) + 2)]
        for label in labels:
            values = ppl.get((calib, label, dataset), {})
            if not values:
                continue
            row = [f'{values[seed]:.6f}' if seed in values else '—'
                   for seed in seeds]
            span = max(values.values()) - min(values.values())
            lines.append(f'| {label} | ' + ' | '.join(row) + f' | {span:.6f} |')
        lines.append('')

    conflicts = {str(seed): record.get('consistency', [])
                 for seed, record in records.items()
                 if record.get('consistency')}
    payload['consistency'] = conflicts
    if conflicts:
        lines += ['### 食い違い', '',
                  '同じ校正・同じ配分を別の評価で組み直した結果が一致しなかった'
                  'seed がある。', '']
        for seed, items in conflicts.items():
            lines.append(f'- seed {seed}: {items}')
    else:
        lines.append('重複して組み直した構成は、どの seed でも塊ごとの NLL まで一致した。')
    heads = {record.get('repository', {}).get('head') for record in records.values()}
    if len(heads) > 1:
        lines += ['', f'**注意**: seed によってコードのコミットが違う（{heads}）。']
    lines.append('')

    report = '\n'.join(lines)
    (out / 'seeds.md').write_text(report, encoding='utf-8')
    with (out / 'seeds.json').open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(report, flush=True)
    print(f'書き出し: {out / "seeds.md"} / {out / "seeds.json"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
