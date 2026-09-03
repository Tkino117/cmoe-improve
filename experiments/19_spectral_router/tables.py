"""本測定の `summary.json` から、report/19 に貼る表を作る。

読むのは `cmoe run` が既に書いたものだけで、モデルは走らせない。区間も再抽出も
CLI が済ませてあるので、ここがするのは並べ替えと整形である。

    uv run python experiments/19_spectral_router/tables.py \
        --run result_logs/spectral_router_bench
"""

import argparse
import json
import os

METRICS = ('acc', 'acc_norm', 'gold_nll', 'margin', 'ref_agreement', 'ref_kl')


def load(path):
    with open(os.path.join(path, 'summary.json')) as handle:
        return json.load(handle)


def interval(row):
    """再抽出の結果を「差 [下, 上] (改善した層/層数)」にする。"""
    if row is None:
        return '—'
    text = (f'{row["mean_difference"]:+.6f} '
            f'[{row["lower"]:+.6f}, {row["upper"]:+.6f}]')
    if row['lower'] > 0 or row['upper'] < 0:
        text = f'**{text}**'
    return f'{text} ({row["improved_strata"]}/{row["n_strata"]})'


def macro_table(payload):
    bench = payload['bench_summary']
    rows = bench['configurations']
    reference = bench.get('reference_macro') or {}
    print(f'対照 = {bench["baseline"]}\n')
    header = '| 指標 | ' + ' | '.join(row['router'] for row in rows) + ' |'
    print(header)
    print('|---|' + '--:|' * len(rows))
    for metric in METRICS:
        cells = []
        for row in rows:
            value = row['macro'].get(metric)
            cells.append('—' if value is None else f'{value:.4f}')
        print(f'| `{metric}` | ' + ' | '.join(cells) + ' |')

    print('\n対照比（(seed × タスク) 層化の問題単位の対応再抽出、95%区間）:\n')
    print('| ルーター | ' + ' | '.join(f'`{m}`' for m in METRICS) + ' |')
    print('|---|' + '---|' * len(METRICS))
    for row in rows[1:]:
        paired = row.get('paired_vs_baseline') or {}
        cells = [interval(paired.get(metric)) for metric in METRICS]
        print(f'| `{row["router"]}` | ' + ' | '.join(cells) + ' |')


def task_table(payload, metric):
    bench = payload['bench_summary']
    rows = bench['configurations']
    tasks = list(rows[0]['tasks'])
    print(f'\n**`{metric}`**\n')
    print('| ルーター | ' + ' | '.join(tasks) + ' |')
    print('|---|' + '--:|' * len(tasks))
    for row in rows:
        cells = [f'{row["tasks"][task][metric]:.4f}' for task in tasks]
        print(f'| `{row["router"]}` | ' + ' | '.join(cells) + ' |')


def ppl_table(payload):
    summary = payload['summary']
    datasets = list(payload['datasets'])
    print(f'\n対照 = {summary["baseline"]}\n')
    print('| ルーター | ' + ' | '.join(datasets) + ' | 対照比 NLL |')
    print('|---|' + '--:|' * len(datasets) + '---|')
    for row in summary['configurations']:
        cells, differences = [], []
        for dataset in datasets:
            entry = row['datasets'].get(dataset, {})
            cells.append(f'{entry.get("mean_ppl", float("nan")):.6f}')
            paired = entry.get('paired_vs_baseline')
            if paired:
                mark = ('**' if paired['lower'] > 0 or paired['upper'] < 0
                        else '')
                differences.append(
                    f'{dataset} {mark}{paired["mean_nll_difference"]:+.6f} '
                    f'[{paired["lower"]:+.6f}, {paired["upper"]:+.6f}]{mark} '
                    f'{paired["improved_seeds"]}/{paired["n_seeds"]}')
        print(f'| `{row["router"]}` | ' + ' | '.join(cells) + ' | '
              + ' / '.join(differences) + ' |')


def load_rows(payload):
    print('\n実際に走った routed expert 数（評価データの上での実測）\n')
    print('| seed | 構成 | wikitext2 | c4-new | bench | bench の分布 (k=0..5) |')
    print('|---|---|--:|--:|--:|---|')
    for run in payload.get('runs', []):
        rows = run.get('realized_load') or {}
        names = sorted({key.split('/')[0] for key in rows
                        if rows[key] is not None})
        for name in names:
            spread = rows.get(f'{name}/bench_histogram')
            def cell(key):
                value = rows.get(f'{name}/{key}')
                return '—' if value is None else f'{value:.4f}'
            text = ('—' if not spread
                    else '[' + ' '.join(f'{v:.3f}' for v in spread) + ']')
            print(f'| {run["seed"]} | `{name}` | {cell("wikitext2")} | '
                  f'{cell("c4-new")} | {cell("bench")} | {text} |')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='result_logs/spectral_router_bench')
    args = parser.parse_args()
    payload = load(args.run)

    print('## PPL')
    ppl_table(payload)
    print('\n## ベンチ マクロ平均')
    macro_table(payload)
    for metric in ('acc', 'acc_norm', 'gold_nll'):
        task_table(payload, metric)
    load_rows(payload)


if __name__ == '__main__':
    main()
