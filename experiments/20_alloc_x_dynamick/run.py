"""20 配分探索 × 可変 Top-K の 2×2 を、校正 seed 3本で測る。

report/07 は層方向の割り付け（層ごとの shared 数 x）を、report/19 はトークン
方向の割り付け（routed の起動個数）を、それぞれ単独で校正データから決めると
良くなることを示した。**両方を同時に入れたときに足し算になるか**を測る。

report/19 の方式7 と方式8 は、軸としては独立だったのに両方入れると単独のどちら
にも届かなかった。「独立な軸だから足せる」はこの系では通らない。

探索は走らせない。report/07 が出した幅4 の配分ベクトルを ``search.json`` から
読むだけである（その seed の探索が無ければ止まる）。

  uv run python experiments/20_alloc_x_dynamick/run.py --smoke   # 2層・8問
  uv run python experiments/20_alloc_x_dynamick/run.py --seed 0  # 約80分
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
NEXPERTS, NACTIVE = 8, 6
WIDTH = 4                     # 読む探索の幅（report/07 の主役）
# 固定の配分。先頭が `cmoe run` の対応のある比較の基準になるので uniform3 を先に
# 置く（report/07 の固定対照であり、report/19 で方式8 を測った動作点でもある）
FIXED_ALLOCS = ('uniform3', 'uniform5')
# 1回の変換の中に2方式を載せる。carve も expert 重みも共有され、差は MoE.gate
# だけになる。伝播に使われるのは先頭の cmoe（report/19 と同じ取り決め）
ROUTERS = 'cmoe,dynamic_cmoe'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
# report/07 の探索の出力。ここから配分ベクトルを読む
SEARCH_DIR = 'result_logs/{calib}_w{width}_n{nsamples}_seed{seed}'

# 結果を変えない引数（出力先・チャンク幅・基準の取り込み元）。同じ実験かの判定から外す
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


def searched_allocation(seed):
    """report/07 が出した幅4 の配分を読む。無ければ止まる（勝手に探索しない）。"""
    out_dir = ROOT / SEARCH_DIR.format(calib=CALIB, width=WIDTH,
                                       nsamples=NSAMPLES, seed=seed)
    payload = out_dir / 'search.json'
    if not payload.exists():
        raise SystemExit(
            f'{payload} が無い。report/07 の探索（幅{WIDTH} / seed {seed}）が要る')
    record = load(payload)
    if 'allocation' not in record:
        raise SystemExit(f'{payload} に配分が入っていない（予算切れの記録か）')
    values = record['allocation']['values']
    if len(values) != 32:
        raise SystemExit(f'{payload} の配分が32層ぶんでない（{len(values)}）')
    return values, record['score']


def bench_stage(seed, root, plan, searched):
    out_dir = root / f'bench_alloc_dynamic_seed{seed}'
    specs = list(plan['fixed_allocs'])
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
    # 可変Kは fit の上でしきい値を決める。既定の 64 系列は report/19 と同じで、
    # smoke ではそこまで引く意味が無い（成分に1本ずつ配れる下限は7）
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

    n_configs = len(specs) * len(ROUTERS.split(','))

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
            'n_configs': n_configs}


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
                        help='2層・8問。経路の確認')
    args = parser.parse_args(argv)

    plan = {
        'layers': 2 if args.smoke else None,
        # slimpajama は7成分に最低1本ずつ配る
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'fixed_allocs': ('uniform3',) if args.smoke else FIXED_ALLOCS,
        'fit_samples': 8 if args.smoke else None,
        # smoke は問題を絞るので report/04 の dense（全件）は取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    root = Path(args.out) if args.out else (
        ROOT / 'result_logs' / ('exp20_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'width': WIDTH, 'routers': ROUTERS.split(','),
              'smoke': args.smoke}

    if args.smoke:
        # smoke は2層しか変換しないので、32層ぶんの配分ベクトルは貼れない
        searched = None
        log(f'seed {args.seed} / 校正 {CALIB} n={plan["nsamples"]}（smoke）')
    else:
        searched, score = searched_allocation(args.seed)
        mean_x = sum(searched) / len(searched)
        zero_k = sum(1 for value in searched if value == NACTIVE)
        record['searched'] = {'values': searched, 'score': score,
                              'mean_x': mean_x, 'mean_k': NACTIVE - mean_x,
                              'zero_k_layers': zero_k}
        log(f'seed {args.seed} / 校正 {CALIB} n={plan["nsamples"]} / '
            f'探索 幅{WIDTH}')
        log(f'  読んだ配分: 平均 x={mean_x:.4f}  平均 K={NACTIVE - mean_x:.3f}  '
            f'K=0 の層 {zero_k}/32')
    log(f'出力 {root}')

    log(f'\n=== 測定 / seed {args.seed} ===')
    record['bench'] = bench_stage(args.seed, root, plan, searched)
    log(f'  {record["bench"]["n_configs"]} 構成')

    record['seconds'] = time.time() - started
    summarize(root / f'exp20_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
