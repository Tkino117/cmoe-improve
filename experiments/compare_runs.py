"""2つの `cmoe run` の出力を、seed で対応づけて比べる。

`cmoe run` の中の比較は「同じ変換設定の中で配分とルーターを振ったもの」しか
扱わない（`--nexperts` は run ごとに1つしか取れない）。粒度 N を跨いだ比較は
run をまたぐので、ここが引き受ける。

再抽出の単位と層は CLI と同じものを使う:

* PPL … seed を層とした、評価塊の対応再抽出
* ベンチ … (seed × タスク) を層とした、問題の対応再抽出

使い方:

    uv run python experiments/compare_runs.py \
        --base result_logs/router5_s3a3e8 --base-alloc uniform3 \
        --cand result_logs/e16_granularity --cand-alloc uniform6

``--base`` / ``--cand`` は繰り返せる。seed ごとに別ディレクトリへ書く実験
（experiments/05・07・09）を1本の比較にまとめるためで、同じ seed が2つの
ディレクトリから出てきたら止まる。
"""

import argparse
import json
import os
from types import SimpleNamespace

from cmoe.eval import bench, bench_stats
from cmoe.eval.stats import paired_differences, stratified_paired_bootstrap

REPS, BOOT_SEED = 10000, 20260813


def load(path):
    with open(os.path.join(path, 'summary.json')) as handle:
        return json.load(handle)


def run_label(run):
    """run を1つの名前で呼ぶ。``cmoe run`` と ``cmoe prune`` の両方を受ける。

    ``prune`` の run には配分もルーターも無い。手法とスパース率の組で呼ぶので、
    ``--cand-alloc flap@block/0.25`` のように指定できる。
    """
    if 'allocation' in run:
        return run['allocation']['name']
    return f'{run["method"]}@{run["scope"]}/{run["sparsity"]:g}'


def run_matches(run, alloc):
    """``--base-alloc`` / ``--cand-alloc`` がこの run を指しているか。

    探索が出した配分の ``name`` はどれも ``custom`` で、1つのディレクトリに
    幅2/3/4 の3本が並ぶことがある。名前だけでは引けないので、**層ごとの値を
    並べたベクトル**でも指せるようにする（``cmoe run --alloc`` に貼れる形と
    同じ書き方である）。
    """
    if run_label(run) == alloc:
        return True
    values = run.get('allocation', {}).get('values')
    return values is not None and ','.join(str(v) for v in values) == alloc


def run_has_router(run, router):
    return 'routers' not in run or router in run['routers']


def run_ppl(run, router, dataset):
    """``run`` は router ごとに割れているが、``prune`` は割れていない。"""
    rows = run['ppl']
    return rows[dataset] if dataset in rows else rows[router][dataset]


def run_bench_path(run, router):
    rows = run['bench_samples']
    return rows if isinstance(rows, str) else rows[router]


def pick(payloads, alloc, router):
    """(seed -> (ディレクトリ, run))。同じ配分・ルーターの run を seed で引く。

    生の尤度のパスは run が書かれたディレクトリからの相対なので、seed ごとに
    ディレクトリが違う実験のために、どこから来た run かを一緒に持つ。
    """
    rows = {}
    for path, payload in payloads.items():
        # ``@last`` は「その実行に渡された最後の --alloc」を指す。探索が出した
        # 配分は seed ごとに違うベクトルなので、ディレクトリを跨いで1つの名前で
        # は指せない。experiments/06・07 は幅2/3/4 をこの順で渡しており、最後が
        # 幅4 である（各ディレクトリの summary.json の arguments で確かめられる）
        wanted = (payload['arguments']['alloc'][-1] if alloc == '@last' else alloc)
        for run in payload['runs']:
            if not run_matches(run, wanted) or not run_has_router(run, router):
                continue
            if run['seed'] in rows:
                raise SystemExit(
                    f'seed {run["seed"]} の {alloc}+{router} が2つある'
                    f'（{rows[run["seed"]][0]} と {path}）')
            rows[run['seed']] = (path, run)
    if not rows:
        raise SystemExit(f'{alloc}+{router} の run が無い')
    return rows


def ppl_table(base, cand, base_router, cand_router, datasets):
    out = {}
    for name in datasets:
        strata, base_ppl, cand_ppl = [], [], []
        for seed in sorted(set(base) & set(cand)):
            left = run_ppl(base[seed][1], base_router, name)
            right = run_ppl(cand[seed][1], cand_router, name)
            base_ppl.append(left['ppl'])
            cand_ppl.append(right['ppl'])
            strata.append(paired_differences(
                SimpleNamespace(chunk_mean_nlls=left['chunk_mean_nlls']),
                SimpleNamespace(chunk_mean_nlls=right['chunk_mean_nlls'])))
        out[name] = {
            'base_mean_ppl': sum(base_ppl) / len(base_ppl),
            'cand_mean_ppl': sum(cand_ppl) / len(cand_ppl),
            'paired': stratified_paired_bootstrap(
                strata, reps=REPS, seed=BOOT_SEED),
        }
    return out


def bench_table(base, cand, base_router, cand_router, reference):
    seeds = sorted(set(base) & set(cand))
    loaded = {
        'base': [bench.load_samples(os.path.join(
            base[s][0], run_bench_path(base[s][1], base_router))) for s in seeds],
        'cand': [bench.load_samples(os.path.join(
            cand[s][0], run_bench_path(cand[s][1], cand_router))) for s in seeds],
    }
    means = {}
    for side in ('base', 'cand'):
        per_seed = [bench_stats.summarize(one, reference) for one in loaded[side]]
        means[side] = {
            'macro': {metric: sum(one['macro'][metric] for one in per_seed)
                      / len(per_seed) for metric in per_seed[0]['macro']},
            'tasks': {task: {metric: sum(one['tasks'][task][metric]
                                         for one in per_seed) / len(per_seed)
                             for metric in per_seed[0]['tasks'][task]}
                      for task in per_seed[0]['tasks']},
        }
    paired = {}
    for metric in means['base']['macro']:
        strata = []
        for left, right in zip(loaded['base'], loaded['cand']):
            for task, samples in right.items():
                strata.append(bench_stats.paired_differences(
                    left[task], samples, metric,
                    reference=reference.get(task) if reference else None))
        result = stratified_paired_bootstrap(
            strata, reps=REPS, seed=BOOT_SEED, key='mean_difference',
            unit='paired-question-within-seed-and-task',
            lower_is_better=not bench_stats.HIGHER_IS_BETTER[metric])
        result['improved_strata'] = result.pop('improved_seeds')
        result['n_strata'] = result.pop('n_seeds')
        paired[metric] = result
    return means, paired, len(seeds)


def show(label, paired, key):
    low, high = paired['lower'], paired['upper']
    mark = '*' if low > 0 or high < 0 else ' '
    strata = paired.get('improved_strata', paired.get('improved_seeds'))
    total = paired.get('n_strata', paired.get('n_seeds'))
    print(f'  {label:<16} {paired[key]:+.6f} '
          f'[{low:+.6f}, {high:+.6f}]{mark} {strata}/{total}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', action='append', required=True,
                        help='対照の run ディレクトリ。繰り返せる')
    parser.add_argument('--base-alloc', required=True)
    parser.add_argument('--cand', action='append', required=True,
                        help='候補の run ディレクトリ。繰り返せる')
    parser.add_argument('--cand-alloc', required=True)
    parser.add_argument('--router', default='cmoe')
    parser.add_argument('--base-router', default=None)
    parser.add_argument('--cand-router', default=None)
    parser.add_argument('--reference', default=None,
                        help='dense の基準 JSON。既定は候補側の bench/dense.json')
    args = parser.parse_args()

    base_payloads = {path: load(path) for path in args.base}
    cand_payloads = {path: load(path) for path in args.cand}
    base_router = args.base_router or args.router
    cand_router = args.cand_router or args.router
    base = pick(base_payloads, args.base_alloc, base_router)
    cand = pick(cand_payloads, args.cand_alloc, cand_router)
    seeds = sorted(set(base) & set(cand))

    base_payload = base_payloads[args.base[0]]
    cand_payload = cand_payloads[args.cand[0]]
    ba, ca = base_payload['arguments'], cand_payload['arguments']
    print(f'対照   {", ".join(args.base)} {args.base_alloc}+{base_router} '
          f'N={ba["nexperts"]} A={ba["nactive"]} 校正 {ba["calib"]}')
    print(f'候補   {", ".join(args.cand)} {args.cand_alloc}+{cand_router} '
          f'N={ca["nexperts"]} A={ca["nactive"]} 校正 {ca["calib"]}')
    print(f'seed   {seeds}')

    datasets = [name for name in base_payload['datasets']
                if all(name in payload['datasets']
                       for payload in cand_payloads.values())]
    print(f'\n== PPL（{len(seeds)} seed 平均。'
          '差は共通評価塊の NLL、負なら候補が良い）==')
    for name, row in ppl_table(base, cand, base_router, cand_router,
                               datasets).items():
        print(f'  {name:<12} 対照 {row["base_mean_ppl"]:.6f} → '
              f'候補 {row["cand_mean_ppl"]:.6f}')
        show('NLL 差', row['paired'], 'mean_nll_difference')

    reference = bench.load_samples(
        args.reference or os.path.join(args.cand[0], 'bench', 'dense.json'))
    means, paired, n_seeds = bench_table(
        base, cand, base_router, cand_router, reference)
    print(f'\n== ベンチ マクロ平均（{n_seeds} seed 平均）==')
    order = ['acc', 'acc_norm', 'gold_nll', 'margin', 'ref_kl', 'ref_agreement']
    print(f'  {"":<16}{"対照":>12}{"候補":>12}')
    for metric in order:
        print(f'  {metric:<16}{means["base"]["macro"][metric]:>12.6f}'
              f'{means["cand"]["macro"][metric]:>12.6f}')
    print('\n== 対照比（* は95%区間が0をまたがない）==')
    for metric in order:
        show(metric, paired[metric], 'mean_difference')

    print('\n== タスクごと（acc / acc_norm / gold_nll、対照 → 候補）==')
    for task in means['base']['tasks']:
        left, right = means['base']['tasks'][task], means['cand']['tasks'][task]
        print(f'  {task:<16} '
              f'{left["acc"]:.4f}→{right["acc"]:.4f}  '
              f'{left["acc_norm"]:.4f}→{right["acc_norm"]:.4f}  '
              f'{left["gold_nll"]:.4f}→{right["gold_nll"]:.4f}')


if __name__ == '__main__':
    main()
