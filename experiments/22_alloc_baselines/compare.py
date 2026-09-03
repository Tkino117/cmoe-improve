"""22-e 対照3本を、提案（beam 幅4）と校正が選んだ一様の両方と比べる。GPU は要らない。

``experiments/compare_runs.py`` は配分を**名前**で引くが、ここに並ぶ配分は
``cmoe run`` の中ではどれも ``custom`` を名乗る。加えて順序対照も LExI も OWL も
**seed ごとに別のベクトル**である。どちらも compare_runs の引き方では表せない
ので、ここが値と seed で引き直す。再抽出の単位・層・回数・seed は compare_runs と
同じものを使う（同じ関数を呼んでいる）ので、出る数字は res09 / res15 / res21 の
表と同じ規則で読める。

差の向きは **候補 − 対照** で、対照は既定で新しい run の側（順序対照・LExI・
OWL）、候補は提案である。つまり **「提案 − 対照」** が出る。

最初に走るのは**再現の検査**である。新しい run に同梱した一様配分と、
``bench_slimpajama*`` の同じ一様配分の数字が一致するかを見る。作業ツリーには
report/19・20・22 で触った変更があるので、これが合わないうちは run をまたぐ
比較を読んではいけない（report/21 と同じ理由）。

  uv run python experiments/22_alloc_baselines/compare.py --nactive 6
  uv run python experiments/22_alloc_baselines/compare.py --nactive 4
  uv run python experiments/22_alloc_baselines/compare.py --nactive 6 --against uniform
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_runs import bench_table, load, ppl_table, show  # noqa: E402

from cmoe.eval import bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
ROUTER = 'cmoe'
METRICS = ('acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement')
# 主表の行。allocate.py が付けた名前に加え、既存の測定を指す擬似的な行が2本
# ある（``uniform`` = 校正が選んだ一様、``ew_default`` = report/21 の EW-rule）。
# 1枚の表に「規則型2本・順序対照3本・LExI・提案」が並ぶようにするためで、
# どちらも新しい測定は伴わない
ROWS = ('ew_default', 'uniform', 'reverse', 'shuffle1', 'shuffle2', 'lexi',
        'owl_default')
PSEUDO_ROWS = ('uniform', 'ew_default')


def spec_of(run):
    return ','.join(str(x) for x in run['allocation']['values'])


def measurements(run):
    """run をまたいで突き合わせる数。同じ配分なら1ビットも違わないはず。"""
    return ({name: run['ppl'][ROUTER][name]['mean_nll']
             for name in run['ppl'][ROUTER]},
            run['bench'][ROUTER]['macro']['acc'])


def pick_by_spec(payloads, specs):
    """(seed -> (ディレクトリ, run))。名前ではなく**値**で引く。

    同じ値の run が2つ出ることはある — LExI の argmin が全層 x=A に落ちれば、
    その配分は新しい run にも既存の ``bench_slimpajama*`` にも
    （``uniform{A}`` として）居る。**そのときは数字が一致することを確かめてから
    先に見つけた方を使う。** 一致しなければ止める: 同じ配分が違う数字を出すのは
    配分の差ではなくコードの差であり、比較そのものが成り立たない。
    """
    rows, seen = {}, {}
    for path, payload in payloads.items():
        for run in payload['runs']:
            seed = run['seed']
            if seed not in specs or spec_of(run) != specs[seed]:
                continue
            if ROUTER not in run['routers']:
                continue
            if seed in rows:
                if measurements(run) != measurements(rows[seed][1]):
                    raise SystemExit(
                        f'seed {seed} の {specs[seed]} が2つあり、数字が違う'
                        f'（{rows[seed][0]} と {path}）')
                seen[seed] = seen.get(seed, [rows[seed][0]]) + [path]
                continue
            rows[seed] = (path, run)
    for seed, paths in sorted(seen.items()):
        print(f'  （seed {seed} の {specs[seed][:20]}… は {len(paths)} 箇所に '
              f'あり、数字は一致。{paths[0]} を使う）')
    missing = sorted(set(specs) - set(rows))
    if missing:
        raise SystemExit(f'seed {missing} の配分 {specs} が見つからない')
    return rows


def reproduction_check(new_payloads, old_payloads, baseline_spec):
    """同じ一様配分が、新旧の run で同じ数字を出すか。

    合わなければ止める。ここが合わないまま先へ進むと、配分の差ではなく
    コードの差を測ったことになる。
    """
    specs = {seed: baseline_spec for seed in SEEDS}
    new = pick_by_spec(new_payloads, specs)
    old = pick_by_spec(old_payloads, specs)
    print(f'\n== 再現の検査（同梱した {baseline_spec[:11]}… を旧 run と突き合わせる）==')
    ok = True
    for seed in SEEDS:
        left, right = old[seed][1], new[seed][1]
        if left['data']['carve']['token_hash'] != right['data']['carve']['token_hash']:
            print(f'  seed {seed}  ** 校正トークンのハッシュが違う **')
            ok = False
            continue
        parts = []
        for name in left['ppl'][ROUTER]:
            gap = (right['ppl'][ROUTER][name]['mean_nll']
                   - left['ppl'][ROUTER][name]['mean_nll'])
            parts.append(f'{name} Δmean_nll={gap:+.3e}')
            ok = ok and gap == 0.0
        gap_acc = (right['bench'][ROUTER]['macro']['acc']
                   - left['bench'][ROUTER]['macro']['acc'])
        parts.append(f'macro acc Δ={gap_acc:+.3e}')
        ok = ok and gap_acc == 0.0
        print(f'  seed {seed}  ' + '  '.join(parts))
    print('  → ' + ('完全一致。run をまたぐ比較を読んでよい' if ok else
                    '** 一致しない。ここから先の比較は読めない **'))
    return ok


def report(title, base, cand, datasets, reference, base_name, cand_name):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')
    print(f'\n== PPL（{len(base)} seed 平均。差は共通評価塊の NLL、'
          f'負なら {cand_name} が良い）==')
    ppl = ppl_table(base, cand, ROUTER, ROUTER, datasets)
    for name, row in ppl.items():
        print(f'  {name:<12} {base_name} {row["base_mean_ppl"]:.6f} → '
              f'{cand_name} {row["cand_mean_ppl"]:.6f}')
        show('NLL 差', row['paired'], 'mean_nll_difference')

    means, paired, n_seeds = bench_table(base, cand, ROUTER, ROUTER, reference)
    print(f'\n== ベンチ マクロ平均（{n_seeds} seed 平均）==')
    print(f'  {"":<16}{base_name:>14}{cand_name:>14}')
    for metric in METRICS:
        print(f'  {metric:<16}{means["base"]["macro"][metric]:>14.6f}'
              f'{means["cand"]["macro"][metric]:>14.6f}')
    print(f'\n== {cand_name} − {base_name}（* は95%区間が0をまたがない）==')
    for metric in METRICS:
        show(metric, paired[metric], 'mean_difference')
    print(f'\n== タスクごと acc（{base_name} → {cand_name}）==')
    for task in means['base']['tasks']:
        left = means['base']['tasks'][task]['acc']
        right = means['cand']['tasks'][task]['acc']
        print(f'  {task:<16} {left:.4f} → {right:.4f}  ({right - left:+.4f})')
    return means, paired, ppl


def uniform_spec(name, n_layers):
    return ','.join([name.replace('uniform', '')] * n_layers)


def row_specs(name, allocs, n_layers, tag):
    """行の名前 → seed ごとの配分ベクトル。

    擬似的な行2本だけは、この実験の候補ではなく既存の測定を指す。
    """
    if name == 'uniform':
        return {s: uniform_spec(allocs[s]['calibration_uniform'], n_layers)
                for s in SEEDS}
    if name == 'ew_default':
        specs = {}
        for s in SEEDS:
            record = json.load(open(
                ROOT / f'result_logs/ew_bench{tag}_seed{s}/ew_rule.json'))
            specs[s] = ','.join(str(x) for x in record['values'])
        return specs
    specs = {}
    for s in SEEDS:
        entry = next((e for e in allocs[s]['candidates']
                      if e['name'] == name or name in e['aliases']), None)
        if entry is None:
            raise SystemExit(f'seed {s} の候補に {name} が無い')
        specs[s] = entry['spec']
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--rows', default=','.join(ROWS))
    parser.add_argument('--against', default='proposal',
                        choices=('proposal', 'uniform'),
                        help='比べる相手。proposal = beam 幅4、'
                             'uniform = 校正が選んだ一様')
    parser.add_argument('--json', default=None, help='数値の書き出し先')
    args = parser.parse_args(argv)

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    rows = [name.strip() for name in args.rows.split(',') if name.strip()]
    new_dirs = [f'result_logs/bench22{tag}_seed{s}' for s in SEEDS]
    old_dirs = [f'result_logs/bench_slimpajama{tag}_seed{s}' for s in SEEDS]
    allocs = {s: json.load(open(ROOT / f'result_logs/alloc22{tag}_seed{s}.json'))
              for s in SEEDS}

    ew_dirs = [f'result_logs/ew_bench{tag}_seed{s}' for s in SEEDS]
    new_payloads = {p: load(p) for p in new_dirs}
    old_payloads = {p: load(p) for p in old_dirs}
    ew_payloads = {p: load(p) for p in ew_dirs if Path(ROOT / p).exists()}
    # 新しい run を先に置く。同じ配分が両方にあるときは新しい方を採る
    both = {**new_payloads, **old_payloads, **ew_payloads}
    datasets = list(next(iter(new_payloads.values()))['datasets'])
    reference = bench.load_samples(
        os.path.join(new_dirs[0], 'bench', 'dense.json'))
    n_layers = len(allocs[SEEDS[0]]['proposal']['values'])

    print(f'スパース率 {100 - 100 * args.nactive // 8}%  N=8 A={args.nactive}  '
          f'校正 slimpajama n=16  seed {list(SEEDS)}')
    proposal_specs = {s: allocs[s]['proposal']['spec'] for s in SEEDS}
    uniform_specs = {s: uniform_spec(allocs[s]['calibration_uniform'], n_layers)
                     for s in SEEDS}
    print('提案（beam 幅4）:')
    for s in SEEDS:
        print(f'  seed{s}  {proposal_specs[s]}')
    print('校正が選んだ一様: '
          + ', '.join(f'seed{s}={allocs[s]["calibration_uniform"]}'
                      for s in SEEDS))

    baseline_spec = uniform_spec(f'uniform{args.nactive // 2}', n_layers)
    if not reproduction_check(new_payloads, old_payloads, baseline_spec):
        raise SystemExit('再現の検査に落ちた')

    if args.against == 'proposal':
        cand_name, cand_specs = '提案', proposal_specs
    else:
        cand_name, cand_specs = '一様', uniform_specs
    cand = pick_by_spec(both, cand_specs)

    collected = {}
    for name in rows:
        specs = row_specs(name, allocs, n_layers, tag)
        base = pick_by_spec(both, specs)
        mean_x = [sum(int(v) for v in specs[s].split(',')) / n_layers
                  for s in SEEDS]
        title = (f'{name} 対 {cand_name}   '
                 f'（{name} の平均 x = '
                 + ', '.join(f'{v:.3f}' for v in mean_x) + '）')
        means, paired, ppl = report(title, base, cand, datasets, reference,
                                    name, cand_name)
        collected[name] = {
            'specs': specs,
            'mean_x': mean_x,
            'base_macro': means['base']['macro'],
            'cand_macro': means['cand']['macro'],
            'base_tasks': means['base']['tasks'],
            'cand_tasks': means['cand']['tasks'],
            'ppl': {dataset: {'base_mean_ppl': row['base_mean_ppl'],
                              'cand_mean_ppl': row['cand_mean_ppl'],
                              **{k: row['paired'][k] for k in
                                 ('mean_nll_difference', 'lower', 'upper',
                                  'improved_seeds', 'n_seeds')}}
                    for dataset, row in ppl.items()},
            'difference': {metric: {k: paired[metric][k] for k in
                                    ('mean_difference', 'lower', 'upper',
                                     'improved_strata', 'n_strata')}
                           for metric in METRICS},
        }

    if args.json:
        with open(args.json, 'w') as handle:
            json.dump({'n_active': args.nactive, 'against': args.against,
                       'seeds': list(SEEDS),
                       'cand_mean_x': [sum(allocs[s]['proposal']['values'])
                                       / n_layers for s in SEEDS]
                       if args.against == 'proposal' else
                       [sum(int(v) for v in uniform_specs[s].split(','))
                        / n_layers for s in SEEDS],
                       'rows': collected}, handle,
                      indent=1, ensure_ascii=False)
        print(f'\n書いた: {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
