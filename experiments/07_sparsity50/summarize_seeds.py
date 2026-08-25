"""07 の複数 seed をまとめる。GPU は使わない。

各 seed の ``bench_slimpajama_a4_seed<N>/summary.json`` と ``bench/*.json``、
それに ``score_uniform_slimpajama_a4_n16_seed<N>/score.json`` を読み、構成ごとに
seed 間の平均と、対照との差を出す。生の対数尤度が残っているので、指標も対応の
ある比較もここで作り直せる（測り直しは要らない）。

  uv run python experiments/07_sparsity50/summarize_seeds.py --seeds 0,1,2

06 との違いは動作点（N=8 / A=4）だけである。一様は x=0..4 の5本になり、固定対照
は A の半分の ``uniform2`` になる。読み方は 06 と同じ。

配分は seed ごとに変わるので、突き合わせる鍵はベクトルではなく「同じ手続きで
作った配分」である。一様は名前で、探索の結果は**幅**で引く。

**対照を3つ並べる。**

1. ``uniform2`` … 固定。選び方に評価指標が入らない
2. **校正で選んだ一様** … 一様5本を探索と同じ ``suffix_kl`` で採点した最良。
   探索も対照も「校正だけを見て選んだ1本」になるので、これが対等な比較である
3. 評価指標で最良の一様 … その指標を見てから5本の最良を取る。対照の側にだけ
   たまたまの上振れが乗るので、探索に不利な側の下限として読む

区間はすべて **seed を層とした対応のあるブートストラップ**で、PPL は評価塊、
ベンチは (seed × タスク) を層とした問題単位である。seed 自体は再抽出しない
（3本から分布は推定できない）。
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import BASELINE, CALIB, NACTIVE, NSAMPLES, ROOT, TAG, WIDTHS  # noqa: E402

from cmoe.eval import bench, bench_stats  # noqa: E402
from cmoe.eval.stats import stratified_paired_bootstrap  # noqa: E402

UNIFORMS = [f'uniform{x}' for x in range(NACTIVE + 1)]
SEARCHED = [f'beam 幅{width}' for width in WIDTHS]
ORDER = UNIFORMS + SEARCHED
METRICS = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
DATASETS = ['wikitext2', 'c4-new']
FIXED_BASELINE = BASELINE
REPS, BOOT_SEED = 10000, 20260813


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def searched_labels(root, seed, nsamples, widths=WIDTHS):
    """探索が出したベクトル -> ラベル。幅が違っても同じ配分に着くことがある。"""
    labels = {}
    for width in widths:
        path = root / f'{CALIB}_{TAG}_w{width}_n{nsamples}_seed{seed}' / 'search.json'
        if not path.exists():
            raise SystemExit(f'{path} が無い')
        record = load(path)
        if 'allocation' not in record:
            raise SystemExit(f'{path} は終わっていない探索である')
        spec = ','.join(str(value) for value in record['allocation']['values'])
        labels.setdefault(spec, []).append(width)
    return {spec: [f'beam 幅{width}' for width in widths]
            for spec, widths in labels.items()}


def load_seed(root, seed, nsamples, widths=WIDTHS):
    """1 seed 分。ラベル -> 指標・生の尤度・塊ごとの NLL、と対照の選ばれ方。"""
    stage = root / f'bench_{CALIB}_{TAG}_seed{seed}'
    payload = stage / 'summary.json'
    if not payload.exists():
        raise SystemExit(f'{payload} が無い')
    record = load(payload)
    if 'bench_summary' not in record or record.get('failures'):
        raise SystemExit(f'{payload} は終わっていない')

    labels = searched_labels(root, seed, nsamples, widths)
    rows, matched = {}, 0
    for run in record['runs']:
        spec = ','.join(str(value) for value in run['allocation']['values'])
        names = labels.get(spec)
        matched += names is not None
        names = names or [run['allocation']['name']]
        relative = run.get('bench_samples', {}).get('cmoe')
        if relative is None:
            raise SystemExit(f'{stage} の {names} にベンチの生データが無い')
        samples = bench.load_samples(stage / relative)
        chunks = {dataset: run['ppl']['cmoe'][dataset]['chunk_mean_nlls']
                  for dataset in DATASETS}
        ppl = {dataset: run['ppl']['cmoe'][dataset]['ppl'] for dataset in DATASETS}
        for name in names:
            rows[name] = {'samples': samples, 'chunks': chunks, 'ppl': ppl,
                          'spec': spec}
    # 探索配分の行が1つも解決しなければ、この実験の主役の比較が表から**黙って
    # 消える**。落ちないぶん気づきにくいので、ここで名指しする
    if labels and not matched:
        print(f'警告: {stage} の配分は探索のベクトルと1つも一致しなかった'
              '（層を切った smoke ならこれで正しい）', file=sys.stderr)
    reference = bench.load_samples(stage / 'bench' / 'dense.json')

    score_path = (root / f'score_uniform_{CALIB}_{TAG}_n{nsamples}_seed{seed}'
                  / 'score.json')
    calib_choice = None
    if score_path.exists():
        scored = load(score_path)
        calib_choice = {'best': scored['best']['name'],
                        'scores': {row['allocation']['name']: row['score']
                                   for row in scored['scores']}}
    else:
        print(f'警告: {score_path} が無い。校正で選んだ対照は出せない',
              file=sys.stderr)
    return {'record': record, 'rows': rows, 'reference': reference,
            'calib': calib_choice}


def spread(values):
    """平均 ± 標準偏差（1本しか無ければ標準偏差は 0）。"""
    if len(values) < 2:
        return sum(values) / len(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def best_uniform(means, metric):
    """その seed の中で、この指標が最も良かった一様配分。"""
    available = [name for name in UNIFORMS if name in means]
    if not available:
        return None
    key = max if bench_stats.HIGHER_IS_BETTER[metric] else min
    return key(available, key=lambda name: means[name]['macro'][metric])


def best_uniform_ppl(loaded, seed, dataset):
    """その seed の中で、この評価セットの PPL が最も低かった一様配分。"""
    rows = loaded[seed]['rows']
    available = [name for name in UNIFORMS if name in rows]
    return min(available, key=lambda name: rows[name]['ppl'][dataset])


def bench_comparison(label, seeds, loaded, means, baseline_of):
    """ベンチ6指標の、対照からの差。``baseline_of(seed, metric)`` が対照を返す。"""
    row = {}
    for metric in METRICS:
        chosen, differences, strata = {}, [], []
        for seed in seeds:
            base = baseline_of(seed, metric)
            if base is None or base not in means[seed]:
                continue
            chosen[seed] = base
            differences.append(means[seed][label]['macro'][metric]
                               - means[seed][base]['macro'][metric])
            reference = loaded[seed]['reference']
            for task, samples in loaded[seed]['rows'][label]['samples'].items():
                strata.append(bench_stats.paired_differences(
                    loaded[seed]['rows'][base]['samples'][task], samples,
                    metric, reference=reference.get(task)))
        if not differences:
            continue
        higher = bench_stats.HIGHER_IS_BETTER[metric]
        mean, sigma = spread(differences)
        row[metric] = {
            'baseline_per_seed': chosen,
            'difference_mean': mean,
            'difference_sd': sigma,
            'improved_seeds': sum(1 for value in differences
                                  if (value > 0) == higher and value != 0),
            'n_seeds': len(differences),
            'paired': stratified_paired_bootstrap(
                strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
                unit='paired-question-within-seed-and-task',
                lower_is_better=not higher),
        }
    return row


def ppl_comparison(label, seeds, loaded, baseline_of):
    """PPL の、対照からの平均 NLL の差。塊単位・seed 層。"""
    row = {}
    for dataset in DATASETS:
        chosen, differences, strata = {}, [], []
        for seed in seeds:
            base = baseline_of(seed, dataset)
            rows = loaded[seed]['rows']
            if base is None or base not in rows:
                continue
            chosen[seed] = base
            left, right = rows[base]['chunks'][dataset], rows[label]['chunks'][dataset]
            if len(left) != len(right):
                raise SystemExit(f'seed {seed} {dataset}: 塊の数が食い違う')
            values = [after - before for before, after in zip(left, right)]
            strata.append(values)
            differences.append(sum(values) / len(values))
        if not differences:
            continue
        mean, sigma = spread(differences)
        row[dataset] = {
            'baseline_per_seed': chosen,
            'difference_mean': mean,
            'difference_sd': sigma,
            'n_seeds': len(differences),
            'paired': stratified_paired_bootstrap(
                strata, reps=REPS, seed=BOOT_SEED),
        }
    return row


def build(root, seeds, nsamples, widths=WIDTHS):
    loaded = {seed: load_seed(root, seed, nsamples, widths) for seed in seeds}
    means = {seed: {label: bench_stats.summarize(row['samples'],
                                                 loaded[seed]['reference'])
                    for label, row in loaded[seed]['rows'].items()}
             for seed in seeds}

    order = UNIFORMS + [f'beam 幅{width}' for width in widths]
    labels = [label for label in order
              if all(label in loaded[seed]['rows'] for seed in seeds)]
    if not labels:
        raise SystemExit('全 seed に揃った配分が1つも無い')

    absolute = {}
    for label in labels:
        absolute[label] = {
            'ppl': {dataset: spread([loaded[seed]['rows'][label]['ppl'][dataset]
                                     for seed in seeds])
                    for dataset in DATASETS},
            'macro': {metric: spread([means[seed][label]['macro'][metric]
                                      for seed in seeds])
                      for metric in METRICS},
            'tasks': {task: {metric: spread(
                          [means[seed][label]['tasks'][task][metric]
                           for seed in seeds])
                      for metric in METRICS}
                      for task in means[seeds[0]][label]['tasks']},
        }

    # 対照3種。どれも「seed ごとに選んでから、その seed の中で引く」
    calib_choice = {seed: (loaded[seed]['calib'] or {}).get('best')
                    for seed in seeds}
    controls = {
        'fixed': {
            'title': f'固定対照（{FIXED_BASELINE}）',
            'bench': lambda seed, metric: FIXED_BASELINE,
            'ppl': lambda seed, dataset: FIXED_BASELINE,
        },
        'calibrated': {
            'title': '校正で選んだ一様（suffix_kl の最良）',
            'bench': lambda seed, metric: calib_choice[seed],
            'ppl': lambda seed, dataset: calib_choice[seed],
        },
        'oracle_uniform': {
            'title': '評価指標で最良の一様',
            'bench': lambda seed, metric: best_uniform(means[seed], metric),
            'ppl': lambda seed, dataset: best_uniform_ppl(loaded, seed, dataset),
        },
    }

    relative = {}
    for name, control in controls.items():
        if name == 'calibrated' and any(
                calib_choice[seed] is None for seed in seeds):
            continue
        relative[name] = {
            'title': control['title'],
            'rows': {label: {
                'bench': bench_comparison(label, seeds, loaded, means,
                                          control['bench']),
                'ppl': ppl_comparison(label, seeds, loaded, control['ppl']),
            } for label in labels if label != FIXED_BASELINE or name != 'fixed'},
        }

    return {
        'seeds': seeds,
        'labels': labels,
        'n_experts': 8,
        'n_active': NACTIVE,
        'allocations': {seed: {label: loaded[seed]['rows'][label]['spec']
                               for label in labels} for seed in seeds},
        'calibration_choice': {
            seed: loaded[seed]['calib'] for seed in seeds},
        'n_docs': means[seeds[0]][labels[0]]['n_docs'],
        'absolute': absolute,
        'relative': relative,
    }


def interval(entry):
    boot = entry['paired']
    mark = '**' if not (boot['lower'] <= 0.0 <= boot['upper']) else ''
    return (f'{entry["difference_mean"]:+.6f} '
            f'{mark}[{boot["lower"]:+.6f}, {boot["upper"]:+.6f}]{mark}')


def render(table, out):
    seeds = table['seeds']
    labels = table['labels']
    lines = [f'# 07 スパース率50%（N=8 / A={NACTIVE}）まとめ'
             f'（seed {", ".join(str(s) for s in seeds)}）',
             '', f'問題数: ' + ' / '.join(f'{task} {n}'
                                          for task, n in table['n_docs'].items()), '']

    lines += ['## 校正で選ばれた一様配分', '',
              f'一様 x=0〜{NACTIVE} を探索と同じ `suffix_kl` で採点した結果。'
              'score は同じ校正トークンの上でしか比べられないので、'
              '**seed をまたいで縦に読まない**。', '',
              '| seed | ' + ' | '.join(UNIFORMS) + ' | 選ばれた |',
              '|---' * (len(UNIFORMS) + 2) + '|']
    for seed in seeds:
        choice = table['calibration_choice'][seed]
        if choice is None:
            lines.append(f'| {seed} | ' + ' | '.join('—' for _ in UNIFORMS) + ' | — |')
            continue
        cells = [f'{choice["scores"][name]:.6e}' if name in choice['scores']
                 else '—' for name in UNIFORMS]
        lines.append(f'| {seed} | ' + ' | '.join(cells) + f' | **{choice["best"]}** |')

    lines += ['', '## 探索が出した配分', '',
              '| seed | 幅 | 平均 x | 配分 |', '|---|---|---|---|']
    for seed in seeds:
        for label in labels:
            if not label.startswith('beam'):
                continue
            spec = table['allocations'][seed][label]
            values = [int(field) for field in spec.split(',')]
            lines.append(f'| {seed} | {label.replace("beam 幅", "")} | '
                         f'{sum(values) / len(values):.4f} | `{spec}` |')

    lines += ['', '## PPL（seed 間の平均 ± 標準偏差）', '',
              '| 配分 | ' + ' | '.join(DATASETS) + ' |',
              '|---' * (len(DATASETS) + 1) + '|']
    for label in labels:
        cells = [f'{table["absolute"][label]["ppl"][d][0]:.6f} ± '
                 f'{table["absolute"][label]["ppl"][d][1]:.6f}' for d in DATASETS]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    lines += ['', '## ベンチ マクロ平均（seed 間の平均 ± 標準偏差）', '',
              '| 配分 | ' + ' | '.join(METRICS) + ' |',
              '|---' * (len(METRICS) + 1) + '|']
    for label in labels:
        cells = [f'{table["absolute"][label]["macro"][m][0]:.6f} ± '
                 f'{table["absolute"][label]["macro"][m][1]:.6f}' for m in METRICS]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    for name, control in table['relative'].items():
        lines += ['', f'## {control["title"]} との差', '',
                  '同じ seed の中で引いてから seed 間で平均した。角括弧は'
                  'ブートストラップ95%区間（PPL は塊単位、ベンチは問題単位・'
                  '(seed × タスク) 層）、太字は0をまたがないもの。'
                  'PPL は負が改善、ベンチの向きは指標ごとに違う。', '']
        lines += ['### PPL', '',
                  '| 配分 | ' + ' | '.join(DATASETS) + ' |',
                  '|---' * (len(DATASETS) + 1) + '|']
        for label, row in control['rows'].items():
            if not row['ppl']:
                continue
            cells = [interval(row['ppl'][d]) if d in row['ppl'] else '—'
                     for d in DATASETS]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines += ['', '### ベンチ', '',
                  '| 配分 | ' + ' | '.join(METRICS) + ' |',
                  '|---' * (len(METRICS) + 1) + '|']
        for label, row in control['rows'].items():
            if not row['bench']:
                continue
            cells = [interval(row['bench'][m]) if m in row['bench'] else '—'
                     for m in METRICS]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        if name == 'oracle_uniform':
            lines += ['', '選ばれた一様（指標 × seed）', '',
                      '| 指標 | ' + ' | '.join(f'seed {s}' for s in seeds) + ' |',
                      '|---' * (len(seeds) + 1) + '|']
            any_label = next(iter(control['rows']))
            for metric in METRICS:
                entry = control['rows'][any_label]['bench'].get(metric)
                if entry is None:
                    continue
                cells = [entry['baseline_per_seed'].get(seed, '—') for seed in seeds]
                lines.append(f'| {metric} | ' + ' | '.join(cells) + ' |')
            lines += ['', '| 評価セット | '
                      + ' | '.join(f'seed {s}' for s in seeds) + ' |',
                      '|---' * (len(seeds) + 1) + '|']
            for dataset in DATASETS:
                entry = control['rows'][any_label]['ppl'].get(dataset)
                if entry is None:
                    continue
                cells = [entry['baseline_per_seed'].get(seed, '—') for seed in seeds]
                lines.append(f'| {dataset} | ' + ' | '.join(cells) + ' |')

    tasks = list(table['n_docs'])
    lines += ['', '## タスクごとの正答率（acc / acc_norm、seed 間の平均）', '',
              '| 配分 | ' + ' | '.join(tasks) + ' | マクロ平均 |',
              '|---' * (len(tasks) + 2) + '|']
    for label in labels:
        row = table['absolute'][label]
        cells = [f'{row["tasks"][t]["acc"][0]:.4f} / {row["tasks"][t]["acc_norm"][0]:.4f}'
                 for t in tasks]
        cells.append(f'{row["macro"]["acc"][0]:.4f} / '
                     f'{row["macro"]["acc_norm"][0]:.4f}')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    out.mkdir(parents=True, exist_ok=True)
    (out / 'seeds.md').write_text('\n'.join(lines), encoding='utf-8')
    with (out / 'seeds.json').open('w') as handle:
        json.dump(table, handle, indent=1, ensure_ascii=False)
    print('\n'.join(lines))
    print(f'\n{out}/seeds.md と seeds.json に書いた')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--root', default=None, help='段の置き場所')
    parser.add_argument('--nsamples', type=int, default=NSAMPLES,
                        help='校正の本数（段のディレクトリ名に入っている）')
    parser.add_argument('--widths', default=','.join(str(w) for w in WIDTHS),
                        help='読む探索の幅')
    parser.add_argument('--out', default=None, help='まとめの書き出し先')
    args = parser.parse_args(argv)

    seeds = [int(field) for field in args.seeds.split(',') if field.strip()]
    root = Path(args.root) if args.root else ROOT / 'result_logs'
    out = Path(args.out) if args.out else root / 'exp07_seeds'
    widths = [int(field) for field in args.widths.split(',') if field.strip()]
    render(build(root, seeds, args.nsamples, widths), out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
