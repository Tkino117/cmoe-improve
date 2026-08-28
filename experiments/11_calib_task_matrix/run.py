"""11 校正タスクと評価タスクの対応（5×5）。

report/11 は「校正を選択問題の train split に替えると効く」ことを示した
（`acc` +0.0361、dense との差の 54%）。だがそこで校正に使ったのは5タスクを
混ぜたものなので、効いているのが**測るタスクそのもの**なのか、それとも
**選択問題らしい形**であって中身は問わないのか、が分かれていない。

ここは校正を1タスクに絞った系を5つ作る。1回の測定は5タスクすべてを採点する
ので、5行を走らせれば 5×5 の表が埋まる。

  H1  各列（評価タスク）で対角（同じタスクで校正した行）が最良 → タスク特異性
  H0  行ごとにほぼ一様に上がる → 効いているのは形であってタスクではない

**校正セット以外は report/11 と変えない。** モデル・動作点・ルーター・評価
セット・ベンチ・ブートストラップ・dense の基準まで同じで、配分も `uniform4`
に固定する（report/11 で、配分を振って動く幅 ±0.003 は校正を替えて動いた
+0.036 の 1/10 以下だった。行ごとに探索すると配分が交絡する）。したがって
探索・採点の段は無く、測定の段だけである。

**ExpertWeaver の校正も1行として並べる。** この軸（校正データを何にするか）
での既存の最良は ExpertWeaver（arXiv:2602.15521）の Flan 多タスク校正なので、
同じモデル・同じトークン量・同じ配分で `flanv2` を走らせる。`--calibs flanv2`
で足す（`cmoe.data.flanv2` に写した経緯と欠けたタスクがある）。

対照の2行は測り直さない。5タスク混合 benchtrain は
`result_logs/bench_benchtrain_seed{0,1,2}`（report/11）、wikitext2 は
`result_logs/bench_h4/wikitext2_seed{0,1,2}`（report/04）の `uniform4` を使う。

  uv run python experiments/11_calib_task_matrix/run.py --smoke   # 2層・8問
  uv run python experiments/11_calib_task_matrix/run.py --seed 0  # 約46分
  uv run python experiments/11_calib_task_matrix/run.py --seed 0 --calibs flanv2
  uv run python experiments/11_calib_task_matrix/run.py --seed 1
  uv run python experiments/11_calib_task_matrix/run.py --seed 2

  # 5×5 の表（GPU 不要）
  uv run python experiments/11_calib_task_matrix/summarize.py --seeds 0,1,2
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from cmoe.cli import build_parser
from cmoe.data.benchtrain import TASKS

ROOT = Path(__file__).resolve().parents[2]

# この実験の唯一の変更点。行1本につき1つ
CALIBS = tuple(f'benchtrain:{task}' for task in TASKS)
# 5×5 の外側に置く行。ExpertWeaver の Flan 多タスク校正を同じ量で写したもの
EXTRA_CALIBS = ('flanv2',)
NSAMPLES = 8
ROUTER = 'cmoe'
# 配分は固定する。校正だけを動かす
ALLOC = 'uniform4'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 動作点は report/04・11 と同じ。スパース率 25%
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


def bench_stage(calib, seed, root, plan):
    """1行ぶん。1校正 × `uniform4` を、PPL と5タスクのベンチで測る。"""
    task = calib.split(':')[-1]
    out_dir = root / f'bench_calibtask_{task}_seed{seed}'
    argv = ['run', '--alloc', ALLOC, '--router', ROUTER, '--seeds', str(seed),
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
    return {'calib': calib, 'dir': str(out_dir.relative_to(ROOT))}


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
                        help='走らせる行（既定は5タスク全部。'
                             f'ほかに {", ".join(EXTRA_CALIBS)} が選べる）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問・両端の2行だけ。経路の確認')
    args = parser.parse_args(argv)

    calibs = ([field.strip() for field in args.calibs.split(',') if field.strip()]
              if args.calibs else list(CALIBS))
    plan = {
        'layers': 2 if args.smoke else None,
        'nsamples': NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke は問題を8問に絞る。report/04 の dense は全件で測ったものなので
        # 取り込めない（``cmoe run`` が limit の違いを見て断る）。自分で測る
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    if args.smoke and not args.calibs:
        # 母集団が最も大きい行と最も小さい行。引きが通るかはここで決まる
        calibs = ['benchtrain:piqa', 'benchtrain:arc_challenge']
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp11_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'配分 {ALLOC} 固定 / 校正 n={plan["nsamples"]} / '
        f'行 {", ".join(calibs)}' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'alloc': ALLOC, 'nsamples': plan['nsamples'],
              'nexperts': NEXPERTS, 'nactive': NACTIVE, 'calibs': calibs,
              'smoke': args.smoke, 'rows': []}
    for calib in calibs:
        log(f'\n=== 測定 {calib} / seed {args.seed} ===')
        record['rows'].append(bench_stage(calib, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp11_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
