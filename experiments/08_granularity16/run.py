"""08 粒度 N=16 / A=12 で、配分探索を seed 3本で回す。

`result_logs/e16_granularity`（report 未執筆）は、同じ 25% スパース率で
N=8/A=6 の S3A3E8 と N=16/A=12 を一様配分だけで比べた。**E=16 では一様しか
試していない** ので、ここで探索を入れる。

**粒度以外は e16_granularity と1つも変えない。** 校正（wikitext2 n=8、seed
0/1/2）・ルーター（現行 CMoE）・分割・評価セット・ベンチ・ブートストラップは
すべて同じで、校正トークンのハッシュが一致することで確かめられる。

対照の取り方は report/07・08 と同じ。**探索と同じオラクルで一様を採点し、
そこで選ばれた1本**を対照にする（`cmoe score`）。両側とも「校正だけを見て
選んだ1本」になる。固定対照は `uniform6`（= A/2、shared と routed が半々で、
A=6 の uniform3 と同じ立ち位置）。

1 seed は3段でできている。**済んだ段は飛ばす**。

  1. 探索  ``cmoe search``            → result_logs/wikitext2_e16a12_w<W>_n8_seed<N>
  2. 採点  ``cmoe score``  一様 x=0..12 → result_logs/score_uniform_wikitext2_e16a12_n8_seed<N>
  3. 測定  ``cmoe run --bench``        → result_logs/bench_wikitext2_e16a12_seed<N>

段3 が測るのは3本だけである — 固定対照 `uniform6`、段2 が選んだ一様、段1 の
探索配分。A=12 では一様が13本あり、全部ベンチに掛けると時間がそこへ消える。
`uniform6` と `uniform12` は e16_granularity で測ってあるので、必要なら
`experiments/compare_runs.py` で引き当てられる。

dense の基準は測らない。report/04 のものを取り込む。

  uv run python experiments/08_granularity16/run.py --smoke
  uv run python experiments/08_granularity16/run.py --seed 0
  uv run python experiments/08_granularity16/run.py --seed 1
  uv run python experiments/08_granularity16/run.py --seed 2
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

CALIB = 'wikitext2'
NSAMPLES = 8
ROUTER = 'cmoe'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 8
# この実験の主題は配分であって粒度ではない。N / A は e16_granularity に揃える
NEXPERTS = 16
NACTIVE = 12
# 動作点が名前に入る。A=6 の出力（report/06・07）と同じディレクトリに落ちると、
# 引数が食い違って ensure が止まる
TAG = f'e{NEXPERTS}a{NACTIVE}'
# 固定対照。A の半分＝ shared と routed が半々で、A=6 の uniform3 と同じ立ち位置
BASELINE = f'uniform{NACTIVE // 2}'
# 段2 で採点する一様。x の小さい順に並べる（表として読むのはこちらの順）
SCORED_UNIFORMS = [f'uniform{x}' for x in range(NACTIVE + 1)]
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
    """動作点。3段すべてに同じものを渡す。ここがこの実験の唯一の変更点である。"""
    return ['--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE)]


def common_oracle_flags(seed, plan):
    argv = ['--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', plan['oracle'], '--seed', str(seed), *sparsity_flags()]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv


def search_stage(seed, width, root, plan):
    """段1。層ごとの x を探す。"""
    out_dir = (root /
               f'{CALIB}_{TAG}_{plan["oracle"]}_w{width}_n{plan["nsamples"]}_seed{seed}')
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
    out_dir = (root / f'score_uniform_{CALIB}_{TAG}_{plan["oracle"]}'
               f'_n{plan["nsamples"]}_seed{seed}')
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


def bench_stage(seed, root, plan, searched, chosen):
    """段3。固定対照・校正が選んだ一様・探索配分を、PPL とベンチマークで測る。

    先頭が ``cmoe run`` の対応のある比較の基準になるので、固定対照を先に置く。
    A=12 では一様が13本あり、全部をベンチに掛けると時間がそこへ消えるので、
    測るのは「校正だけを見て選んだ1本」どうしと固定対照の3本に絞る。
    """
    out_dir = root / f'bench_{CALIB}_{TAG}_seed{seed}'
    specs = [BASELINE] + ([chosen] if chosen and chosen != BASELINE else [])
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
    parser.add_argument('--widths', default='2')
    parser.add_argument('--oracle', default='local_error',
                        help='採点オラクル。安い順に mass / local_error / suffix_kl')
    parser.add_argument('--stages', default='search,score,bench',
                        help='走らせる段（既定は全部）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・2本・8問。経路の確認')
    args = parser.parse_args(argv)

    widths = [int(field) for field in args.widths.split(',') if field.strip()]
    stages = [field.strip() for field in args.stages.split(',') if field.strip()]
    plan = {
        'layers': 2 if args.smoke else None,
        'nsamples': 2 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'oracle': args.oracle,
        # smoke で見たいのは経路であって表ではない
        'scored_uniforms': [BASELINE, f'uniform{NACTIVE}'] if args.smoke
                           else SCORED_UNIFORMS,
        # smoke は問題を8問に絞る。report/04 の dense は全件で測ったものなので
        # 取り込めない（``cmoe run`` が limit の違いを見て断る）。自分で測る
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    if args.smoke:
        widths = widths[:1]
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp08_smoke'
    # 相対パスで渡されても out_dir.relative_to(ROOT) が通るように絶対化する
    root = Path(args.out).resolve() if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 {CALIB} n={plan["nsamples"]} / オラクル {plan["oracle"]} / '
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
        if 'score' not in record:
            record['score'] = score_stage(args.seed, root, plan)
        record['bench'] = bench_stage(
            args.seed, root, plan, record['search'],
            record['score']['best']['name'])

    record['seconds'] = time.time() - started
    summarize(root / f'exp08_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
