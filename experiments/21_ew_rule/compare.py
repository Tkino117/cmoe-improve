"""21-d EW-rule を、対照2本と比べる。GPU は要らない。

``experiments/compare_runs.py`` は配分を**名前**で引くが、EW-rule もビーム探索も
``cmoe run`` の中では ``custom`` を名乗るので、名前では区別できない。加えて
「校正が選んだ一様」は **seed ごとに別の配分**である（25% は uniform6/5/6）。
どちらも compare_runs の引き方では表せないので、ここが値と seed で引き直す。

再抽出の単位・層・回数・seed は compare_runs と同じものを使う（同じ関数を
呼んでいる）ので、出る数字は res09 / res15 の表と同じ規則で読める。

  uv run python experiments/21_ew_rule/compare.py --nactive 6
  uv run python experiments/21_ew_rule/compare.py --nactive 4
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


def spec_of(run):
    return ','.join(str(x) for x in run['allocation']['values'])


def pick_by_spec(payloads, specs):
    """(seed -> (ディレクトリ, run))。``specs`` は seed -> 配分ベクトル。

    名前ではなく**値**で引く。同じ値の run が2つあれば止める（compare_runs の
    名前引きと同じ約束）。
    """
    rows = {}
    for path, payload in payloads.items():
        for run in payload['runs']:
            seed = run['seed']
            if seed not in specs or spec_of(run) != specs[seed]:
                continue
            if ROUTER not in run['routers']:
                continue
            if seed in rows:
                raise SystemExit(
                    f'seed {seed} の {specs[seed]} が2つある'
                    f'（{rows[seed][0]} と {path}）')
            rows[seed] = (path, run)
    missing = sorted(set(specs) - set(rows))
    if missing:
        raise SystemExit(f'seed {missing} の配分が見つからない')
    return rows


def report(title, base, cand, datasets, reference):
    print(f'\n{"=" * 72}\n{title}\n{"=" * 72}')
    print('\n== PPL（3 seed 平均。差は共通評価塊の NLL、負なら EW-rule が良い）==')
    for name, row in ppl_table(base, cand, ROUTER, ROUTER, datasets).items():
        print(f'  {name:<12} 対照 {row["base_mean_ppl"]:.6f} → '
              f'EW-rule {row["cand_mean_ppl"]:.6f}')
        show('NLL 差', row['paired'], 'mean_nll_difference')

    means, paired, n_seeds = bench_table(base, cand, ROUTER, ROUTER, reference)
    order = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
    print(f'\n== ベンチ マクロ平均（{n_seeds} seed 平均）==')
    print(f'  {"":<16}{"対照":>12}{"EW-rule":>12}')
    for metric in order:
        print(f'  {metric:<16}{means["base"]["macro"][metric]:>12.6f}'
              f'{means["cand"]["macro"][metric]:>12.6f}')
    print('\n== 対照比（* は95%区間が0をまたがない）==')
    for metric in order:
        show(metric, paired[metric], 'mean_difference')
    print('\n== タスクごと acc（対照 → EW-rule）==')
    for task in means['base']['tasks']:
        left = means['base']['tasks'][task]['acc']
        right = means['cand']['tasks'][task]['acc']
        print(f'  {task:<16} {left:.4f} → {right:.4f}  ({right - left:+.4f})')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nactive', type=int, default=6)
    parser.add_argument('--against', default=None,
                        help='この一様配分とだけ比べる（例 uniform3）。'
                             '**平均 x を揃えた対照**を作るためにある — '
                             'EW-rule の負けが規則の質なのか、既定 α が出す '
                             '動作点なのかは、これでしか分けられない')
    args = parser.parse_args(argv)

    tag = '' if args.nactive == 6 else f'_a{args.nactive}'
    exp = 'exp06_seeds' if args.nactive == 6 else 'exp07_seeds'
    ew_dirs = [f'result_logs/ew_bench{tag}_seed{s}' for s in SEEDS]
    old_dirs = [f'result_logs/bench_slimpajama{tag}_seed{s}' for s in SEEDS]

    seeds_json = json.load(open(ROOT / f'result_logs/{exp}/seeds.json'))
    allocations = seeds_json['allocations']
    choice = seeds_json['calibration_choice']

    ew_specs = {}
    for s in SEEDS:
        record = json.load(open(ROOT / f'result_logs/ew_bench{tag}_seed{s}'
                                / 'ew_rule.json'))
        ew_specs[s] = ','.join(str(x) for x in record['values'])

    # 対照1: 校正が選んだ一様。seed ごとに違う
    uniform_specs = {s: allocations[str(s)][choice[str(s)]['best']]
                     for s in SEEDS}
    # 対照2: ビーム幅4。labels の並びの最後が幅4
    beam_label = seeds_json['labels'][-1]
    beam_specs = {s: allocations[str(s)][beam_label] for s in SEEDS}

    ew_payloads = {p: load(p) for p in ew_dirs}
    old_payloads = {p: load(p) for p in old_dirs}
    cand = pick_by_spec(ew_payloads, ew_specs)
    datasets = list(next(iter(ew_payloads.values()))['datasets'])
    reference = bench.load_samples(
        os.path.join(ew_dirs[0], 'bench', 'dense.json'))

    print(f'スパース率 {100 - 100 * args.nactive // 8}%  N=8 A={args.nactive}  '
          f'校正 slimpajama n=16  seed {list(SEEDS)}')
    print(f'EW-rule 配分（seed ごと）:')
    for s in SEEDS:
        print(f'  seed{s}  {ew_specs[s]}')
    print(f'校正が選んだ一様: '
          + ', '.join(f'seed{s}={choice[str(s)]["best"]}' for s in SEEDS))

    if args.against:
        specs = {s: allocations[str(s)][args.against] for s in SEEDS}
        report(f'EW-rule 対 {args.against}',
               pick_by_spec(old_payloads, specs), cand, datasets, reference)
        return 0

    report(f'EW-rule 対 校正が選んだ一様',
           pick_by_spec(old_payloads, uniform_specs), cand, datasets, reference)
    report(f'EW-rule 対 探索 {beam_label}',
           pick_by_spec(old_payloads, beam_specs), cand, datasets, reference)
    return 0


if __name__ == '__main__':
    sys.exit(main())
