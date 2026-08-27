"""09 校正データをベンチマークの train split に替える。

report/04 は同じ動作点（N=8 / A=6、スパース率 25%）で wikitext2 と c4 の2つの
校正を測っている。ここはそこへ**校正セットだけが違う3本目**を足す。校正に使う
のは選択問題5タスクの train split（``cmoe.data.benchtrain``）で、答え部分の
マスクは入れない — 「テキストの中身を測る対象に近づけるだけで何か動くか」を
見る。

**校正以外は report/04 と変えない。** モデル・動作点・分割・ルーター・オラクル・
評価セット・ベンチ・ブートストラップはすべて同じである。変えるのは ``--calib``
と、探索の幅（1,2,3）・並べる一様（x=4,5,6）だけで、後者は report/04 の
2,3,4 / x=0..6 と重なる範囲でしか比べない。

評価に使う split には1問も触らないので、report/04 の結果も dense の基準も
そのまま比較対象になる。

1 seed は3段でできている。**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅1 / 2 / 3  → result_logs/benchtrain_w<W>_n8_seed<N>
  2. 採点  ``cmoe score``  一様 x=4,5,6 → result_logs/score_uniform_benchtrain_n8_seed<N>
  3. 測定  ``cmoe run --bench`` 6構成   → result_logs/bench_benchtrain_seed<N>

dense の基準は測らない。report/04 のものを取り込む（dense は変換していない
モデルなので、校正にも seed にも配分にも依らない）。

  uv run python experiments/09_bench_calibration/run.py --smoke   # 2層・8問
  uv run python experiments/09_bench_calibration/run.py --seed 0  # 約5時間
  uv run python experiments/09_bench_calibration/run.py --seed 1
  uv run python experiments/09_bench_calibration/run.py --seed 2

  # この校正の中でのまとめ（GPU 不要）
  uv run python experiments/09_bench_calibration/summarize_seeds.py --seeds 0,1,2

  # report/04 の wikitext2 校正との対応のある比較（GPU 不要）
  uv run python experiments/compare_runs.py \
      --base result_logs/bench_h4/wikitext2_seed0 \
      --base result_logs/bench_h4/wikitext2_seed1 \
      --base result_logs/bench_h4/wikitext2_seed2 --base-alloc uniform4 \
      --cand result_logs/bench_benchtrain_seed0 \
      --cand result_logs/bench_benchtrain_seed1 \
      --cand result_logs/bench_benchtrain_seed2 --cand-alloc uniform4
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

# この実験の唯一の変更点
CALIB = 'benchtrain'
NSAMPLES = 8
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
WIDTHS = (1, 2, 3)
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 動作点は report/04 と同じ。スパース率 25%
NEXPERTS = 8
NACTIVE = 6
# 対照。先頭が cmoe run の対応のある比較の基準になるので固定対照を先に置く。
# 探索が出す配分の平均 x が 4 前後なので、x=4 が量として釣り合う一様である
BASELINE = 'uniform4'
UNIFORMS = [BASELINE, 'uniform5', 'uniform6']
# 段2 で採点する一様。x の小さい順に並べる（表として読むのはこちらの順）
SCORED_UNIFORMS = ['uniform4', 'uniform5', 'uniform6']
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
    """段を1つ通す。済んでいれば飛ばし、書きかけなら退避してから引き直す。

    完了印はファイルの存在ではない（開始直後から真になる）。段ごとに
    ``is_finished`` が中身を見て決める。
    """
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


def sparsity_flags():
    """動作点。3段すべてに同じものを渡す。report/04 と同じ値である。"""
    return ['--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE)]


def common_oracle_flags(seed, plan):
    argv = ['--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', ORACLE, '--seed', str(seed), *sparsity_flags()]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv


def search_stage(seed, width, root, plan):
    """段1。層ごとの x を探す。"""
    out_dir = root / f'{CALIB}_w{width}_n{plan["nsamples"]}_seed{seed}'
    argv = (['search', *common_oracle_flags(seed, plan),
             '--search', 'beam', '--width', str(width),
             '--out', str(out_dir)])

    def is_finished(record):
        # 予算切れの記録が残っていても 'allocation' は入らない。配分が出た実行
        # だけを済みとする
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    return {'width': width, 'dir': str(out_dir.relative_to(ROOT)),
            'values': record['allocation']['values'],
            'score': record['score'],
            'recheck_gap': record['recheck']['gap']}


def score_stage(seed, root, plan):
    """段2。一様配分を、探索と同じオラクルで採点する。"""
    out_dir = root / f'score_uniform_{CALIB}_n{plan["nsamples"]}_seed{seed}'
    argv = ['score', *common_oracle_flags(seed, plan)]
    for spec in plan['scored_uniforms']:
        argv += ['--alloc', spec]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return ('best' in record
                and len(record.get('scores', [])) == len(plan['scored_uniforms']))

    record = ensure(out_dir, 'score.json', argv, is_finished)
    return {'dir': str(out_dir.relative_to(ROOT)),
            'best': record['best'],
            'scores': {row['allocation']['name']: row['score']
                       for row in record['scores']}}


def bench_stage(seed, root, plan, searched):
    """段3。一様3種 + 探索3種を、PPL とベンチマークで測る。"""
    out_dir = root / f'bench_{CALIB}_seed{seed}'
    specs = list(plan['uniforms'])
    labels = {}
    for record in searched:
        spec = ','.join(str(value) for value in record['values'])
        # 幅が違っても同じ配分に着くことがある。行を消さずに束ねる
        labels[spec] = (f'{labels[spec]}/{record["width"]}' if spec in labels
                        else f'beam 幅{record["width"]}')
        specs.append(spec)
    specs = list(dict.fromkeys(specs))

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']), *sparsity_flags(),
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
                and len(record.get('runs', [])) == len(specs))

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'dir': str(out_dir.relative_to(ROOT)), 'labels': labels,
            'specs': specs}


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
    parser.add_argument('--widths', default=','.join(str(w) for w in WIDTHS))
    parser.add_argument('--stages', default='search,score,bench',
                        help='走らせる段（既定は全部）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・5本・8問。経路の確認')
    args = parser.parse_args(argv)

    widths = [int(field) for field in args.widths.split(',') if field.strip()]
    stages = [field.strip() for field in args.stages.split(',') if field.strip()]
    plan = {
        'layers': 2 if args.smoke else None,
        # benchtrain は5タスクに最低1本ずつ配る。smoke でもそれ未満には落とせない
        'nsamples': 5 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke で見たいのは経路であって表ではない。PPL は評価セット全体を走るので
        # 構成数がそのまま時間になる
        'uniforms': [BASELINE, 'uniform6'] if args.smoke else UNIFORMS,
        'scored_uniforms': [BASELINE, 'uniform6'] if args.smoke
                           else SCORED_UNIFORMS,
        # smoke は問題を8問に絞る。report/04 の dense は全件で測ったものなので
        # 取り込めない（``cmoe run`` が limit の違いを見て断る）。自分で測る
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    if args.smoke:
        widths = widths[:1]
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp09_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 {CALIB} n={plan["nsamples"]} / '
        f'幅 {",".join(str(w) for w in widths)} / 段 {",".join(stages)}'
        + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'nexperts': NEXPERTS, 'nactive': NACTIVE,
              'widths': widths, 'smoke': args.smoke}

    if 'search' in stages:
        record['search'] = []
        for width in widths:
            log(f'\n=== 探索 幅{width} / seed {args.seed} ===')
            record['search'].append(search_stage(args.seed, width, root, plan))
            got = record['search'][-1]
            log(f'  score {got["score"]:.6e}  測り直しの差 {got["recheck_gap"]:.3e}')

    if 'score' in stages:
        log(f'\n=== 一様の採点 / seed {args.seed} ===')
        record['score'] = score_stage(args.seed, root, plan)
        best = record['score']['best']
        log(f'  校正で選んだ対照: {best["name"]}（score {best["score"]:.6e}）')

    if 'bench' in stages:
        log(f'\n=== 測定 / seed {args.seed} ===')
        if 'search' not in record:
            record['search'] = [search_stage(args.seed, width, root, plan)
                                for width in widths]
        record['bench'] = bench_stage(args.seed, root, plan, record['search'])

    record['seconds'] = time.time() - started
    summarize(root / f'exp09_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
