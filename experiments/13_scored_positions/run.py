"""13 校正を「採点に効く位置」だけで読む。

report/11 は校正データの中身を評価に近づけて `acc` を +0.036 動かした（dense
との差の 54%）。効いた軸は「どのトークンで活性を数えるか」だったが、動かした
のはトークンの**出どころ**だけである。同じ校正データの中で、**どの位置を
数えるか**はまだ動かしていない。

評価が実際に読むのは、1問を独立に走らせたときの、続きのトークンの対数尤度
だけである。ここはそこへ合わせにいく。変えるのは2つで、分けずに一度に入れる。

  1. 1問1系列（他の問題が文脈に入らない）
  2. 活性を数えるのは、採点される対数尤度を作っている位置だけ

  H1  マクロ `acc` が report/11 の5タスク混合を上回る → 位置を寄せる線は生きる
  H0  上回らない → この線は落とし、タスク特化の主張（docs/04）へ資源を移す

**校正セットと読む位置以外は report/11・12 と変えない。** モデル・動作点・
ルーター・評価セット・ベンチ・ブートストラップ・dense の基準まで同じで、配分も
`uniform4` に固定する。

**量は採点位置で揃える。** `benchqa` は `--nsamples × --seqlen` を**採点位置の
総数**として読むので、既定の 8 × 2048 は report/11 の 16,384 トークンと同じ数の
位置になる。推定に入る量を対照と揃えておかないと、負けたときに「位置が効か
ない」のか「統計が痩せただけ」なのかが分からない。

**対照は測り直さない。** 5タスク混合 benchtrain は
`result_logs/bench_benchtrain_seed{0,1,2}`（report/11）の `uniform4` を使う。

**見ておくところ。** 1問1系列にすると系列数が千の桁になるので、
`capture_layer_inputs` が層0 の入力を一度に作る。`--batch-chunk` はそのあとの
層まわしにしか効かない。落ちるならそこである。

  uv run python experiments/13_scored_positions/run.py --smoke   # 2層・8問
  uv run python experiments/13_scored_positions/run.py --seed 0  # 見積り15分
  uv run python experiments/13_scored_positions/run.py --seed 1
  uv run python experiments/13_scored_positions/run.py --seed 2

  # report/11 の5タスク混合との、対応のある比較
  uv run python experiments/compare_runs.py \\
      --base result_logs/bench_benchtrain_seed0 \\
      --base result_logs/bench_benchtrain_seed1 \\
      --base result_logs/bench_benchtrain_seed2 --base-alloc uniform4 \\
      --cand result_logs/bench_scored_seed0 \\
      --cand result_logs/bench_scored_seed1 \\
      --cand result_logs/bench_scored_seed2 --cand-alloc uniform4
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

# 行の定義。(校正セット, 活性を数える位置)
ROWS = {
    # この実験の1本。1問1系列で引き、採点に効く位置だけを数える
    'scored': ('benchqa', 'scored'),
    # 分解の行。同じ引き方で全位置を数える（形の効果だけ）。既定では走らせない
    'qa_all': ('benchqa', 'all'),
}
DEFAULT_ROWS = ('scored',)
NSAMPLES = 8
ROUTER = 'cmoe'
# 配分は固定する。校正の読み方だけを動かす
ALLOC = 'uniform4'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 動作点は report/04・11・12 と同じ。スパース率 25%
NEXPERTS = 8
NACTIVE = 6
# 1問1系列だと系列が千の桁になるので、層まわしはバッチを分けて進める
BATCH_CHUNK = 128
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


def bench_stage(row, seed, root, plan):
    """1行ぶん。1校正 × `uniform4` を、PPL と5タスクのベンチで測る。"""
    calib, positions = ROWS[row]
    out_dir = root / f'bench_{row}_seed{seed}'
    argv = ['run', '--alloc', ALLOC, '--router', ROUTER, '--seeds', str(seed),
            '--calib', calib, '--nsamples', str(plan['nsamples']),
            '--profile-positions', positions,
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--batch-chunk', str(BATCH_CHUNK),
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
    return {'row': row, 'calib': calib, 'positions': positions,
            'dir': str(out_dir.relative_to(ROOT))}


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
    parser.add_argument('--rows', default=None,
                        help=f'走らせる行（既定は {", ".join(DEFAULT_ROWS)}。'
                             f'ほかに {", ".join(sorted(set(ROWS) - set(DEFAULT_ROWS)))}）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問。経路の確認')
    args = parser.parse_args(argv)

    rows = ([field.strip() for field in args.rows.split(',') if field.strip()]
            if args.rows else list(DEFAULT_ROWS))
    unknown = [row for row in rows if row not in ROWS]
    if unknown:
        raise SystemExit(f'知らない行 {unknown}。{sorted(ROWS)} から選ぶ')

    plan = {
        'layers': 2 if args.smoke else None,
        'nsamples': 1 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke は問題を8問に絞るので、全件で測った dense は取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp13_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'配分 {ALLOC} 固定 / 採点位置 {plan["nsamples"] * 2048} / '
        f'行 {", ".join(rows)}' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'alloc': ALLOC, 'nsamples': plan['nsamples'],
              'nexperts': NEXPERTS, 'nactive': NACTIVE, 'rows': [],
              'smoke': args.smoke}
    for row in rows:
        log(f'\n=== 測定 {row} / seed {args.seed} ===')
        record['rows'].append(bench_stage(row, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp13_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
