"""24 配分探索を方式7 に揃えて回す（25%）。

experiments/23 は report/07 の配分（**方式1 のルーターで採点されたもの**）に
方式7 を後から載せた。そこで見えたのは足し算にならないどころか、探索配分の上
では方式7 の効きが半分以下に落ちるという形である（seed 0）。読み方は2つある。

* 配分探索とルーター改善が同じ予算を奪い合っている（探索配分は平均 K=1.59 で、
  32層中9層が K=0。そこでは方式7 のルーターが載らない）
* 配分が方式1 向きに選ばれているだけで、方式7 の前提で選び直せば話が変わる

**この実験は2つ目を潰す。** 探索の候補の層に方式7 のルーターを載せて採点し、
出てきた配分の上で測り直す。配分がほとんど動かなければ1つ目が残り、大きく動いて
効きが戻るなら結論はひっくり返る。

3層の予備では配分が大きく動いた（方式1 で `6,6,5`・平均 x=5.67、方式7 で
`3,5,4`・平均 x=4.00）。ルーティングが上手くなったぶん、探索が routed を増やす
方向へ動いている。

段は2つ。探索の引数は report/07 と ``--router`` 以外すべて同じにしてある。

  uv run python experiments/24_matched_search/run.py --smoke
  uv run python experiments/24_matched_search/run.py --seed 0
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from cmoe.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]

CALIB = 'slimpajama'
NSAMPLES = 16
NEXPERTS, NACTIVE = 8, 6      # 25%（A=6）。50% は 25% の結果を見てから
WIDTH = 4
ORACLE = 'suffix_kl'
# 探索の候補の層に載せるルーター。rank は名前に添える（探索に --spectral-rank は無い）
SEARCH_ROUTER = 'spectral_mass:128'
# 測るときのルーター。**先頭が層に載り次層へ伝播する**ので cmoe を外せない
ROUTERS = 'cmoe,spectral_mass:128'
FIXED_ALLOC = 'uniform3'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
SEARCH_OUT = 'result_logs/{calib}_spectral128_w{width}_n{nsamples}_seed{seed}'

NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck', 'bench_reference', 'bench_cache_dir'}


def log(message=''):
    print(message, flush=True)


def run_cli(argv):
    command = ['uv', 'run', 'cmoe', *argv]
    log(f'$ {" ".join(command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def retire(path):
    """書きかけのディレクトリを退避する。消さない（落ちた跡は原因調べに要る）。"""
    for index in itertools.count(1):
        target = path.with_name(f'{path.name}.partial{index}')
        if not target.exists():
            path.rename(target)
            log(f'  書きかけの {path.name} を {target.name} へ退避した')
            return target


def same_experiment(record, argv):
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def search_stage(seed, plan):
    """方式7 を載せた探索。既に終わっていれば読むだけ。"""
    out_dir = ROOT / SEARCH_OUT.format(calib=CALIB, width=WIDTH,
                                       nsamples=plan['nsamples'], seed=seed)
    argv = ['search', '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--seed', str(seed), '--width', str(WIDTH), '--oracle', ORACLE,
            '--router', SEARCH_ROUTER]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    payload = out_dir / 'search.json'
    record = load(payload) if payload.exists() else None
    if record is not None and 'allocation' in record:
        log(f'  {out_dir.name}: 済み')
    else:
        if out_dir.exists():
            retire(out_dir)
        run_cli(argv)
        record = load(payload) if payload.exists() else None
        if record is None or 'allocation' not in record:
            raise SystemExit(f'{out_dir} に配分が残らなかった')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    return record['allocation']['values'], record['score']


def bench_stage(seed, root, plan, searched):
    """uniform3 と、方式7 で探索した配分。どちらも2方式を1回の変換で載せる。"""
    out_dir = root / f'bench_matched_a{NACTIVE}_seed{seed}'
    specs = [FIXED_ALLOC]
    if searched is not None:
        specs.append(','.join(str(value) for value in searched))
    specs = list(dict.fromkeys(specs))

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTERS, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
             '--datasets', DATASETS,
             '--bench', '--bench-batch-size', str(plan['bench_batch'])]
    if plan['fit_samples'] is not None:
        argv += ['--fit-samples', str(plan['fit_samples'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    if plan['reference'] is not None:
        reference = ROOT / plan['reference']
        if not reference.exists():
            raise SystemExit(f'{reference} が無い。dense の基準が要る')
        argv += ['--bench-reference', str(reference)]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return (not record.get('failures')
                and 'summary' in record and 'bench_summary' in record
                and len(record.get('runs', [])) == len(specs))

    payload = out_dir / 'summary.json'
    record = load(payload) if payload.exists() else None
    if record is not None and is_finished(record):
        log(f'  {out_dir.name}: 済み')
    else:
        if out_dir.exists():
            if record is not None:
                differences = same_experiment(record, argv)
                if differences:
                    raise SystemExit(
                        f'{out_dir} は違う実験の出力である'
                        f'（{", ".join(differences)}）')
            retire(out_dir)
        run_cli(argv)
        record = load(payload) if payload.exists() else None
        if record is None or not is_finished(record):
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    return {'dir': str(out_dir.relative_to(ROOT)), 'specs': specs,
            'n_configs': len(specs) * len(ROUTERS.split(','))}


def summarize(path, payload):
    payload = dict(payload)
    payload.update({
        'commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        'dirty': bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='3層・8問。経路の確認')
    args = parser.parse_args(argv)

    plan = {
        'layers': 3 if args.smoke else None,
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'fit_samples': 8 if args.smoke else None,
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    root = Path(args.out) if args.out else (
        ROOT / 'result_logs' / ('exp24_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'width': WIDTH, 'nactive': NACTIVE,
              'search_router': SEARCH_ROUTER, 'routers': ROUTERS.split(','),
              'smoke': args.smoke}
    log(f'seed {args.seed} / 校正 {CALIB} n={plan["nsamples"]} / '
        f'探索 幅{WIDTH} × {SEARCH_ROUTER}')
    log(f'出力 {root}')

    log(f'\n=== 段1 探索（方式7 を載せて採点）/ seed {args.seed} ===')
    searched, score = search_stage(args.seed, plan)
    mean_x = sum(searched) / len(searched)
    zero_k = sum(1 for value in searched if value == NACTIVE)
    record['searched'] = {'values': searched, 'score': score,
                          'mean_x': mean_x, 'mean_k': NACTIVE - mean_x,
                          'zero_k_layers': zero_k, 'n_layers': len(searched)}
    log(f'  配分: 平均 x={mean_x:.4f}  平均 K={NACTIVE - mean_x:.3f}  '
        f'K=0 の層 {zero_k}/{len(searched)}  score {score:.6e}')

    if args.smoke and len(searched) != 32:
        # 接頭辞は run --alloc に貼れない
        log('  smoke なので段2 は uniform3 だけで回す')
        searched = None

    log(f'\n=== 段2 測定 / seed {args.seed} ===')
    record['bench'] = bench_stage(args.seed, root, plan, searched)
    log(f'  {record["bench"]["n_configs"]} 構成')

    record['seconds'] = time.time() - started
    summarize(root / f'exp24_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
