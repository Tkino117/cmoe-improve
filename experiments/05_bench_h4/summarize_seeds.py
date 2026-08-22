"""05 の複数 seed をまとめる。GPU は使わない。

各段の ``summary.json`` と ``bench/*.json`` を読み、構成ごとに seed 間の平均と
標準偏差を出す。生の対数尤度が残っているので、指標も対応のある比較もここで
作り直せる（測り直しは要らない）。

  uv run python experiments/05_bench_h4/summarize_seeds.py --seeds 0,1,2

配分は seed ごとに変わるので、突き合わせる鍵はベクトルではなく「同じ手続きで
作った配分」である。一様は名前で、探索の結果は**幅**で引く。校正が変われば
モデルの組み方も変わるので、鍵は（校正, ラベル）になる。

出す表は3つ。

1. マクロ平均（6指標）を seed 3本の平均 ± 標準偏差で
2. タスクごとの ``acc`` / ``acc_norm``（論文の表と同じ形）
3. **同じ seed の中で最良の一様配分から引いた差**。生の標準偏差には、その seed が
   引いた8本の当たり外れが全構成に共通で乗るので、差を見るときはそれを先に
   打ち消す。あわせて、問題単位・(seed × タスク) 層のブートストラップ区間も出す
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import ROOT, WIDTHS, searched_vector  # noqa: E402

from cmoe.eval import bench, bench_stats  # noqa: E402
from cmoe.eval.stats import stratified_paired_bootstrap  # noqa: E402

CALIBS = ('wikitext2', 'c4')
UNIFORMS = [f'uniform{x}' for x in range(7)]
SEARCHED = [f'beam 幅{width}' for width in WIDTHS]
ORDER = UNIFORMS + SEARCHED
METRICS = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
REPS, BOOT_SEED = 10000, 20260813


def labels_for(calib, seed):
    """配分ベクトル -> ラベル。幅が違っても同じ配分に着くことがある。"""
    labels = {}
    for width in WIDTHS:
        spec = searched_vector(calib, width, seed)
        labels.setdefault(spec, []).append(width)
    return {spec: [f'beam 幅{width}' for width in widths]
            for spec, widths in labels.items()}


def load_stage(out, calib, seed):
    """1段を読む。ラベル -> （その構成のタスク別 TaskSamples）と dense 基準。"""
    stage_dir = out / f'{calib}_seed{seed}'
    payload = stage_dir / 'summary.json'
    if not payload.exists():
        raise SystemExit(f'{payload} が無い')
    with payload.open() as handle:
        record = json.load(handle)
    if 'bench_summary' not in record or record.get('failures'):
        raise SystemExit(f'{payload} は終わっていない')

    searched = labels_for(calib, seed)
    rows, matched = {}, 0
    for run in record['runs']:
        spec = ','.join(str(value) for value in run['allocation']['values'])
        names = searched.get(spec)
        matched += names is not None
        names = names or [run['allocation']['name']]
        relative = run.get('bench_samples', {}).get('cmoe')
        if relative is None:
            raise SystemExit(f'{stage_dir} の {names} にベンチの生データが無い')
        samples = bench.load_samples(stage_dir / relative)
        for name in names:
            rows[name] = samples
    # 探索配分の行が1つも解決しなければ、この実験の主役の比較が表から
    # **黙って消える**。落ちないぶん気づきにくいので、ここで名指しする
    if searched and not matched:
        print(f'警告: {stage_dir} の配分は report/03 の探索ベクトルと1つも一致'
              'しなかった（層を切った smoke ならこれで正しい）', file=sys.stderr)
    reference = bench.load_samples(stage_dir / 'bench' / 'dense.json')
    return record, rows, reference


def means_of(samples, reference):
    """1構成の、タスクごとと macro の指標。"""
    return bench_stats.summarize(samples, reference)


def spread(values):
    """平均 ± 標準偏差（1本しか無ければ標準偏差は 0）。"""
    if len(values) < 2:
        return sum(values) / len(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def best_uniform(per_label, metric):
    """その seed の中で、この指標が最も良かった一様配分。"""
    available = [name for name in UNIFORMS if name in per_label]
    if not available:
        return None
    key = (max if bench_stats.HIGHER_IS_BETTER[metric] else min)
    return key(available, key=lambda name: per_label[name]['macro'][metric])


def build(out, seeds):
    """（校正, ラベル）ごとに、seed 間の平均と差を作る。"""
    loaded = {}
    for calib in CALIBS:
        for seed in seeds:
            loaded[(calib, seed)] = load_stage(out, calib, seed)

    tables = {}
    for calib in CALIBS:
        per_seed_means, per_seed_samples, references = {}, {}, {}
        for seed in seeds:
            _, rows, reference = loaded[(calib, seed)]
            per_seed_means[seed] = {name: means_of(samples, reference)
                                    for name, samples in rows.items()}
            per_seed_samples[seed] = rows
            references[seed] = reference

        absolute, relative = {}, {}
        for label in ORDER:
            if not all(label in per_seed_means[seed] for seed in seeds):
                continue
            absolute[label] = {
                'macro': {metric: spread([per_seed_means[seed][label]['macro'][metric]
                                          for seed in seeds])
                          for metric in METRICS},
                'tasks': {task: {metric: spread(
                              [per_seed_means[seed][label]['tasks'][task][metric]
                               for seed in seeds])
                          for metric in METRICS}
                          for task in per_seed_means[seeds[0]][label]['tasks']},
            }
            if label in UNIFORMS:
                continue
            relative[label] = compare_to_best_uniform(
                label, seeds, per_seed_means, per_seed_samples, references)
        if not absolute:
            raise SystemExit(f'{calib}: 全 seed に揃った配分が1つも無い')
        any_label = next(iter(absolute))
        tables[calib] = {'absolute': absolute, 'relative': relative,
                         'n_docs': per_seed_means[seeds[0]][any_label]['n_docs']}
    return tables


def compare_to_best_uniform(label, seeds, per_seed_means, per_seed_samples,
                            references):
    """同じ seed の中で最良の一様配分から引いた差と、その区間。

    引き算を seed の中で済ませてから seed 間で平均する。生の値の標準偏差には
    その seed が引いた8本の当たり外れが**全構成に共通で**乗っており、配分どうし
    の比較にはその成分が要らない。
    """
    row = {}
    for metric in METRICS:
        chosen, differences, strata = {}, [], []
        for seed in seeds:
            base = best_uniform(per_seed_means[seed], metric)
            if base is None:
                continue
            chosen[seed] = base
            differences.append(per_seed_means[seed][label]['macro'][metric]
                               - per_seed_means[seed][base]['macro'][metric])
            for task, samples in per_seed_samples[seed][label].items():
                strata.append(bench_stats.paired_differences(
                    per_seed_samples[seed][base][task], samples, metric,
                    reference=references[seed].get(task)))
        if not differences:
            continue
        higher = bench_stats.HIGHER_IS_BETTER[metric]
        mean, sigma = spread(differences)
        boot = stratified_paired_bootstrap(
            strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
            unit='paired-question-within-seed-and-task', lower_is_better=not higher)
        row[metric] = {
            'baseline_per_seed': chosen,
            'macro_difference_mean': mean,
            'macro_difference_sd': sigma,
            'improved_seeds': sum(1 for value in differences
                                  if (value > 0) == higher and value != 0),
            'n_seeds': len(differences),
            'paired': boot,
        }
    return row


def render(tables, seeds, out):
    """人が読む表。数値は丸めずに JSON にも残す。"""
    lines = [f'# 05 H4 まとめ（seed {", ".join(str(s) for s in seeds)}）', '']
    for calib, table in tables.items():
        docs = ' / '.join(f'{task} {n}' for task, n in table['n_docs'].items())
        lines += [f'## 校正 {calib}', '', f'問題数: {docs}', '',
                  '### マクロ平均（seed 間の平均 ± 標準偏差）', '',
                  '| 配分 | ' + ' | '.join(METRICS) + ' |',
                  '|---' * (len(METRICS) + 1) + '|']
        for label, row in table['absolute'].items():
            cells = [f'{row["macro"][m][0]:.6f} ± {row["macro"][m][1]:.6f}'
                     for m in METRICS]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines += ['', '### 最良の一様配分との差（同じ seed の中で引いてから平均）', '',
                  '負なら「小さいほど良い」指標で改善。括弧は改善した seed の本数、'
                  '角括弧は問題単位・(seed × タスク) 層のブートストラップ95%区間', '',
                  '| 配分 | 指標 | 差 | 区間 | seed |',
                  '|---|---|---|---|---|']
        for label, row in table['relative'].items():
            for metric, entry in row.items():
                boot = entry['paired']
                lines.append(
                    f'| {label} | {metric} | '
                    f'{entry["macro_difference_mean"]:+.6f} ± '
                    f'{entry["macro_difference_sd"]:.6f} | '
                    f'[{boot["lower"]:+.6f}, {boot["upper"]:+.6f}] | '
                    f'{entry["improved_seeds"]}/{entry["n_seeds"]} |')
        tasks = list(table['n_docs'])
        lines += ['', '### タスクごとの正答率（acc / acc_norm、平均 ± 標準偏差）', '',
                  '| 配分 | ' + ' | '.join(tasks) + ' |',
                  '|---' * (len(tasks) + 1) + '|']
        for label, row in table['absolute'].items():
            cells = [f'{row["tasks"][t]["acc"][0]:.4f} / {row["tasks"][t]["acc_norm"][0]:.4f}'
                     for t in tasks]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    (out / 'seeds.md').write_text('\n'.join(lines), encoding='utf-8')
    with (out / 'seeds.json').open('w') as handle:
        json.dump({'seeds': seeds, 'tables': tables}, handle, indent=1,
                  ensure_ascii=False)
    print('\n'.join(lines))
    print(f'\n{out}/seeds.md と seeds.json に書いた')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else ROOT / 'result_logs/bench_h4'
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    render(build(out, seeds), seeds, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
