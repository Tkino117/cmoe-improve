"""26 slimpajama 校正で幅1（貪欲）を測る。

[res15](../../report/icassp/res15-layer-interaction.md) の (b)「候補を1本に絞ると
目的関数も `acc` も最も悪くなる」の根拠は、`benchtrain` 校正の1系しかない。
`result_logs/` に `*_w1_*` は `benchtrain_w1_n8_seed{0,1,2}` だけである。
ここは**校正を slimpajama n=16 に替えて幅1 を足す**。幅2・3・4 は report/07 に
3 seed 揃っているので、新しく測るのは幅1 の1本だけ。

  H1  幅1 が 3 seed とも幅2/3/4 より悪い（校正を替えても順位が変わらない）
  H0  変わる

**動作点・校正・ルーター・評価は report/07 と同一**で、動かしたのは幅だけである。

段は2つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅1     → result_logs/slimpajama_w1_n16_seed<N>
  2. 測定  ``cmoe run --bench``    → result_logs/bench_slimpajama_greedy_seed<N>

段2 は3構成を**同じ実行の中で**測る。先頭の一様は既存の
`bench_slimpajama_seed<N>` との**再現の検査**で、これが Δ=0 にならないうちは
run をまたぐ比較を読んではいけない（report/22 と同じ理由）。

  uv run python experiments/26_greedy_slimpajama/run.py --smoke   # 2層・7本・8問
  uv run python experiments/26_greedy_slimpajama/run.py --seed 0  # 約1.0時間
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

# 動かすのはここだけ。校正も動作点も report/07 と同じ
WIDTH = 1
CONTROL_WIDTH = 4          # 対照。report/07 の主役の幅
CALIB = 'slimpajama'
NSAMPLES = 16
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
NEXPERTS = 8
NACTIVE = 6
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# report/04 が測った dense。取り込む（dense は校正にも seed にも配分にも依らない）
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

# 結果を変えない引数。同じ実験かの判定から外す
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'no_recheck', 'bench_reference', 'bench_cache_dir'}

# 対照（report/07 の幅4）を突き合わせる鍵。**記録にある鍵だけでは足りない**ので
# ここを明示する。あの探索は 1e6ddc7 で測ってあり、そのあと CLI に足した引数
# （`--lookahead` / `--profile-positions`）が記録に無い。無いのは「その頃の CLI に
# 無かった」ということで、今の既定値そのものの振る舞いで測ってある。全鍵を
# 突き合わせると、その2つだけで必ず食い違って止まる。
CONTROL_IDENTITY = ('model', 'adapter', 'calib', 'nsamples', 'seed', 'oracle',
                    'carver', 'nexperts', 'nactive', 'k_act', 'bias_speed',
                    'seqlen', 'search', 'width', 'layers')


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
                        f'（{", ".join(differences)}）。別の --out を指定すること')
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


def suffix(nactive):
    """出力名の動作点。A=6 は無印（report/05〜07）、それ以外は _a<A>（report/09）。"""
    return '' if nactive == NACTIVE else f'_a{nactive}'


def search_dir(root, seed, width, plan):
    return root / (f'{CALIB}{suffix(plan["nactive"])}_w{width}'
                   f'_n{plan["nsamples"]}_seed{seed}')


def search_argv(seed, width, out_dir, plan):
    argv = ['search', '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(plan['nactive']),
            '--search', 'beam', '--width', str(width)]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv + ['--out', str(out_dir)]


def search_stage(seed, root, plan):
    """段1。幅1 で探す。幅1のビームは各層で親の argmin を取るので、貪欲そのもの。"""
    out_dir = search_dir(root, seed, WIDTH, plan)
    argv = search_argv(seed, WIDTH, out_dir, plan)

    def is_finished(record):
        # 予算切れの記録が残っていても 'allocation' は入らない
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    allocation = record['allocation']
    log(f'  幅{WIDTH}: score {record["score"]:.6e} '
        f'平均 x={allocation["mean_x"]:.3f} コスト {record["spent"]:g} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分) '
        f'測り直しの差 {record["recheck"]["gap"]:.3e}')
    log(f'  {allocation["values"]}')
    return {'stage': 'search', 'width': WIDTH,
            'allocation': allocation['values'], 'mean_x': allocation['mean_x'],
            'score': record['score'], 'spent': record['spent'],
            'calls': record['calls'], 'seconds': record['seconds'],
            'recheck_gap': record['recheck']['gap'],
            'dir': str(out_dir.relative_to(ROOT))}


def control_allocation(seed, root, plan):
    """対照（幅4）の配分。measure し直さず report/07 の探索結果を読む。

    smoke には対照が無いので、その場で幅4 を走らせるのではなく一様を置く
    （smoke で見たいのは配線であって、比較ではない）。
    """
    if plan['layers']:
        return None
    path = search_dir(root, seed, CONTROL_WIDTH, plan)
    if not (path / 'search.json').exists():
        raise SystemExit(
            f'{path} が無い。対照になる幅{CONTROL_WIDTH} の探索が要る')
    record = load(path / 'search.json')
    wanted = vars(build_parser().parse_args(
        search_argv(seed, CONTROL_WIDTH, path, plan)))
    given = record.get('arguments', {})
    missing = [key for key in CONTROL_IDENTITY if key not in given]
    if missing:
        raise SystemExit(
            f'{path} の記録に {", ".join(missing)} が無い。'
            '突き合わせるべき鍵が欠けている探索は対照に使えない')
    differences = [key for key in CONTROL_IDENTITY if given[key] != wanted[key]]
    if differences:
        raise SystemExit(
            f'{path} は同じ設定の幅{CONTROL_WIDTH} の探索ではない'
            f'（{", ".join(differences)}）')
    return record['allocation']['values'], record['score']


def bench_stage(candidate, control, seed, root, plan):
    """段2。再現の検査の一様・対照・候補を、同じ実行の中で測る。"""
    out_dir = root / (f'bench_{CALIB}_greedy{suffix(plan["nactive"])}'
                      f'_seed{seed}')
    specs = [plan['recheck_uniform']] + ([] if control is None else [control])
    specs.append(candidate)
    specs = list(dict.fromkeys(specs))

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(plan['nactive']),
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
    return {'stage': 'bench', 'specs': specs,
            'recheck': plan['recheck_uniform'],
            'control': control, 'candidate': candidate,
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
    parser.add_argument('--nactive', type=int, default=NACTIVE,
                        help='1トークンあたりに走る expert 数 A（6 なら 25%%、'
                             '4 なら 50%%）')
    parser.add_argument('--no-bench', action='store_true',
                        help='段1（探索）だけで止める')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・7本・8問。経路の確認')
    args = parser.parse_args(argv)

    plan = {
        'nactive': args.nactive,
        'layers': 2 if args.smoke else None,
        # slimpajama は7成分に最低1本ずつ配る。smoke でもそれ未満には落とせない
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # report/22 と同じ取り方。A=6 なら uniform3、A=4 なら uniform2
        'recheck_uniform': f'uniform{args.nactive // 2}',
        # smoke は問題を8問に絞る。report/04 の dense は全件なので取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    root = Path(args.out) if args.out else (
        ROOT / 'result_logs' / ('exp26_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=True)

    sparsity = 100 * (NEXPERTS - plan['nactive']) // NEXPERTS
    log(f'seed {args.seed} / N={NEXPERTS} A={plan["nactive"]}'
        f'（スパース率 {sparsity}%）/ 校正 {CALIB} n={plan["nsamples"]} / '
        f'幅 {WIDTH}（貪欲）' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'width': WIDTH, 'control_width': CONTROL_WIDTH,
              'nexperts': NEXPERTS, 'nactive': plan['nactive'],
              'stages': [], 'smoke': args.smoke}

    log(f'\n=== 段1 探索 幅{WIDTH} / seed {args.seed} ===')
    stage = search_stage(args.seed, root, plan)
    record['stages'].append(stage)
    candidate = ','.join(str(x) for x in stage['allocation'])

    if not args.no_bench:
        found = control_allocation(args.seed, root, plan)
        if found is None:
            control = None
            log('\n=== 段2 測定（smoke。対照は置かない） ===')
        else:
            values, score = found
            control = ','.join(str(x) for x in values)
            log(f'\n=== 段2 測定 / seed {args.seed} ===')
            log(f'  対照 幅{CONTROL_WIDTH}: score {score:.6e}')
        record['stages'].append(
            bench_stage(candidate, control, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp26_stages{suffix(plan["nactive"])}'
                     f'_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
