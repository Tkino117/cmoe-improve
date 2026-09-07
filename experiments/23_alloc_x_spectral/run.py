"""23 配分探索 × 低ランク score の 2×2 を、2つのスパース率で測る。

report/07・08 は**層方向**の割り付け（層ごとの shared 数 x）を、report/19 は
**score の作り方**（expert の活性質量を低ランクで見積もる方式7）を、それぞれ単独で
校正データから決めると良くなることを示した。**両方を同時に入れたときに
足し算になるか**を測る。

先例がある。report/19 の方式7（低ランク score）と方式8（可変K）は軸としては
独立だったのに、両方入れると単独のどちらにも届かなかった。「独立な軸だから
足せる」はこの系では通らない。experiments/20 は同じ問いを方式8 で測っている。

探索は走らせない。report/07・08 が出した幅4 の配分ベクトルを ``search.json`` から
読むだけである（その seed の探索が無ければ止まる）。**配分は方式1 のルーターで
採点されたものである** — 探索は ``assemble.layer_factory`` が常に基準ルーターを
載せる設計で、方式7 の前提では探索していない。この食い違いは保守的な向きに効く
（配分は方式7 に有利には作られていない）。

ルーターは ``cmoe,spectral_mass:128`` の順で並べる。**先頭が層に載り、次層への
伝播もそれで行われる**（``assemble.py:17``）ので、cmoe を外すと carve そのものが
既存の run と変わってしまい、比較が「gate だけの差」でなくなる。

  uv run python experiments/23_alloc_x_spectral/run.py --smoke
  uv run python experiments/23_alloc_x_spectral/run.py --sparsity 25 --seed 0
  uv run python experiments/23_alloc_x_spectral/run.py --sparsity 50 --seed 0
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
NEXPERTS = 8
WIDTH = 4                     # 読む探索の幅（report/07・08 の主役）

# スパース率ごとの動作点。``uniform`` は既存の bench_slimpajama{,_a4} の対照と
# 同じものを選ぶ（A の半分を shared に置く一様配分）。``suffix`` は探索の出力先
OPERATING_POINTS = {
    25: {'nactive': 6, 'uniform': 'uniform3', 'suffix': ''},
    50: {'nactive': 4, 'uniform': 'uniform2', 'suffix': '_a4'},
}

# 先頭が `cmoe run` の対応のある比較の基準になり、層に載って次層へ伝播もする
ROUTERS = 'cmoe,spectral_mass:128'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
# report/07・08 の探索の出力。ここから配分ベクトルを読む
SEARCH_DIR = 'result_logs/{calib}{suffix}_w{width}_n{nsamples}_seed{seed}'

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


def searched_allocation(seed, point):
    """report/07・08 が出した幅4 の配分を読む。無ければ止まる（勝手に探索しない）。"""
    out_dir = ROOT / SEARCH_DIR.format(calib=CALIB, suffix=point['suffix'],
                                       width=WIDTH, nsamples=NSAMPLES,
                                       seed=seed)
    payload = out_dir / 'search.json'
    if not payload.exists():
        raise SystemExit(
            f'{payload} が無い。幅{WIDTH} / seed {seed} の探索が要る')
    record = load(payload)
    if 'allocation' not in record:
        raise SystemExit(f'{payload} に配分が入っていない（予算切れの記録か）')
    values = record['allocation']['values']
    if len(values) != 32:
        raise SystemExit(f'{payload} の配分が32層ぶんでない（{len(values)}）')
    return values, record['score']


def bench_stage(seed, root, plan, point, searched):
    out_dir = root / f'bench_alloc_spectral_a{point["nactive"]}_seed{seed}'
    specs = list(plan['fixed_allocs'])
    if searched is not None:
        specs.append(','.join(str(value) for value in searched))
    specs = list(dict.fromkeys(specs))

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTERS, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(point['nactive']),
             '--datasets', DATASETS,
             '--bench', '--bench-batch-size', str(plan['bench_batch'])]
    # 方式7 は fit の上で白色化の基底と silu の行の重みを作る。既定の 64 系列は
    # report/19 と同じで、smoke ではそこまで引く意味が無い（下限は7）
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
    parser.add_argument('--sparsity', type=int, default=25,
                        choices=sorted(OPERATING_POINTS),
                        help='スパース率 %%。25 は A=6、50 は A=4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問。経路の確認')
    args = parser.parse_args(argv)
    point = OPERATING_POINTS[args.sparsity]

    plan = {
        'layers': 2 if args.smoke else None,
        # slimpajama は7成分に最低1本ずつ配る
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'fixed_allocs': (point['uniform'],),
        'fit_samples': 8 if args.smoke else None,
        # smoke は問題を絞るので report/04 の dense（全件）は取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    root = Path(args.out) if args.out else (
        ROOT / 'result_logs' / ('exp23_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record = {'seed': args.seed, 'sparsity': args.sparsity,
              'nactive': point['nactive'], 'calib': CALIB,
              'nsamples': plan['nsamples'], 'width': WIDTH,
              'routers': ROUTERS.split(','), 'smoke': args.smoke}

    if args.smoke:
        # smoke は2層しか変換しないので、32層ぶんの配分ベクトルは貼れない
        searched = None
        log(f'seed {args.seed} / 校正 {CALIB} n={plan["nsamples"]}（smoke）')
    else:
        searched, score = searched_allocation(args.seed, point)
        mean_x = sum(searched) / len(searched)
        zero_k = sum(1 for value in searched if value == point['nactive'])
        record['searched'] = {'values': searched, 'score': score,
                              'mean_x': mean_x,
                              'mean_k': point['nactive'] - mean_x,
                              'zero_k_layers': zero_k}
        log(f'seed {args.seed} / スパース率 {args.sparsity}% (A={point["nactive"]}) / '
            f'校正 {CALIB} n={plan["nsamples"]} / 探索 幅{WIDTH}')
        log(f'  読んだ配分: 平均 x={mean_x:.4f}  '
            f'平均 K={point["nactive"] - mean_x:.3f}  K=0 の層 {zero_k}/32')
    log(f'出力 {root}')

    log(f'\n=== 測定 / seed {args.seed} ===')
    record['bench'] = bench_stage(args.seed, root, plan, point, searched)
    log(f'  {record["bench"]["n_configs"]} 構成')

    record['seconds'] = time.time() - started
    summarize(root / f'exp23_stages_a{point["nactive"]}_seed{args.seed}.json',
              record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
