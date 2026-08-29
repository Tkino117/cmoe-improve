"""12 対角の上で shared/routed 配分を振る（S3A3E8 / S5A1E8 / S6A0E8）。

report/12 は校正 × 評価の対角が効くことを示したが、配分は `uniform4`
（= S4A2E8）1点に固定していた。ここは**対角の上だけ**で x を振る。

  x=6 (S6A0E8)  Top-K=0。ルーティングが消え、常時オンの静的マスク
                — すなわち枝刈りの退化ケースである
  x=5 (S5A1E8)  routed 3 個から 1 個
  x=3 (S3A3E8)  CMoE 論文の既定構成
  x=4 (S4A2E8)  report/12 で測り済み。ここでは測らない

**問いは「最適点が内側にあるか」である。** x=6 が最良なら、この動作点で
MoE を選ぶ理由（枝刈りに対する優位）は無い。内点に山があって初めて、
ルーティングが払っている分を回収できていることになる。

先行研究はどちらも端を測っていない。CMoE (arXiv:2502.04416) は既定が
S3A3E8 で Fig 6 が S6A6E16 / S3A9E16 を比べるが A0 は無い。ExpertWeaver
(arXiv:2602.15521) は shared 割合 α を 0.2〜0.7 で掃引しており α=1.0 に
届かない。

**校正セット・動作点・ルーター・評価・dense の基準は report/12 と同じ。**
動かすのは配分だけで、行（校正タスク）は対角の4本に絞る。ARC は
report/12 と同じく1タスクとして扱う（`benchtrain:arc`）。

  uv run python experiments/12_shared_ratio_diagonal/run.py --smoke   # 2層・8問
  uv run python experiments/12_shared_ratio_diagonal/run.py --seed 0  # 約2時間
  uv run python experiments/12_shared_ratio_diagonal/run.py --seed 1
  uv run python experiments/12_shared_ratio_diagonal/run.py --seed 2

  # S3〜S6 × 4タスクの表（GPU 不要）
  uv run python experiments/12_shared_ratio_diagonal/summarize.py --seeds 0,1,2
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

# 対角の4行。report/12 の 4×4 と同じ並びで、ARC は束ねた1行
CALIBS = ('benchtrain:piqa', 'benchtrain:winogrande', 'benchtrain:arc',
          'benchtrain:hellaswag')
# この実験の唯一の変更点。uniform4 は report/12 で測り済みなので入れない
ALLOCS = ('uniform3', 'uniform5', 'uniform6')
NSAMPLES = 8
ROUTER = 'cmoe'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 動作点は report/04・11・12 と同じ。スパース率 25%
NEXPERTS = 8
NACTIVE = 6
# report/04 が測った dense。取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

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
    """記録された引数が、いま走らせようとしている実験と同じものか。"""
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def ensure(out_dir, result_name, argv, is_finished):
    """段を1つ通す。済んでいれば飛ばし、書きかけなら退避してから引き直す。"""
    payload = out_dir / result_name
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
                        f'（{", ".join(differences)}）。別の --out を指定するか、'
                        'そのディレクトリを退けること')
            retire(out_dir)
        run_cli(argv)
        record = load(payload) if payload.exists() else None
        if record is None or not is_finished(record):
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    return record


def bench_stage(calib, alloc, seed, root, plan):
    """対角の1マス。1校正 × 1配分を、PPL と5タスクのベンチで測る。"""
    task = calib.split(':')[-1]
    out_dir = root / f'bench_shared_{task}_{alloc}_seed{seed}'
    argv = ['run', '--alloc', alloc, '--router', ROUTER, '--seeds', str(seed),
            '--calib', calib, '--nsamples', str(plan['nsamples']),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--datasets', DATASETS,
            '--bench', '--bench-batch-size', str(plan['bench_batch'])]
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
                and len(record.get('runs', [])) == 1)

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'calib': calib, 'alloc': alloc, 'dir': str(out_dir.relative_to(ROOT))}


def summarize(path, payload):
    """段の一覧と、測ったコードのコミットを残す。数値そのものは集計しない。"""
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
    parser.add_argument('--calibs', default=None,
                        help='走らせる行（既定は対角の4本）')
    parser.add_argument('--allocs', default=None,
                        help=f'走らせる配分（既定は {", ".join(ALLOCS)}）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問・1マスだけ。経路の確認')
    args = parser.parse_args(argv)

    calibs = ([field.strip() for field in args.calibs.split(',') if field.strip()]
              if args.calibs else list(CALIBS))
    allocs = ([field.strip() for field in args.allocs.split(',') if field.strip()]
              if args.allocs else list(ALLOCS))
    plan = {
        'layers': 2 if args.smoke else None,
        'nsamples': NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    if args.smoke:
        # 端の配分1つだけ。Top-K=0 の経路が通るかはここで決まる
        if not args.calibs:
            calibs = ['benchtrain:piqa']
        if not args.allocs:
            allocs = ['uniform6']
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp12_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 n={plan["nsamples"]} / 配分 {", ".join(allocs)} / '
        f'行 {", ".join(calibs)}' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'allocs': allocs, 'nsamples': plan['nsamples'],
              'nexperts': NEXPERTS, 'nactive': NACTIVE, 'calibs': calibs,
              'smoke': args.smoke, 'cells': []}
    for alloc in allocs:
        for calib in calibs:
            log(f'\n=== 測定 {calib} / {alloc} / seed {args.seed} ===')
            record['cells'].append(
                bench_stage(calib, alloc, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp12_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
