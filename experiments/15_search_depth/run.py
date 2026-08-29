"""15 ビームの深さを1段増やす（幅は増やさない）。

これまでの探索はどれも「1層展開して刈る」だった。刈る時点で見えているのは、
その層を足したときのスコアだけである。層 ℓ の選択が層 ℓ+1 に何を残すかは、
刈ったあとにしか分からない。

ここは幅を固定したまま**深さ**を1段増やす（`--lookahead 1`）。子1つごとに次の層の
候補を全部展開し、**その最良**を子の順位に使う。層 ℓ と層 ℓ+1 の依存を見てから
刈ることになる。

  H1  同じ幅で、深さ1 の配分がオラクルでも `acc` でも幅0 を上回る
  H0  上回らない

**動作点は report/07・09 と同じ。** report/09 でスパース率25%と50%を並べたときの
校正がこの slimpajama n=16 で、seed 0 には幅2・3・4 の探索とベンチが揃っている。
ここは**幅2・深さ1**の1本を足し、幅2・深さ0（`result_logs/slimpajama_w2_n16_seed0`、
score 0.183640 / 平均 x=4.75 / `acc` 0.5763）と比べる。

**コスト。** 採点回数が候補数の1乗ぶん増える。幅2・深さ0 が 7,168 layer_forwards
（2,110秒）だったのに対し、深さ1 は約 56,000（7.8倍、見積り4.6時間）。3層のスモーク
（`result_logs/lookahead_smoke/`）で 5.1倍を実測した。

段は2つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅2・深さ1 → result_logs/slimpajama_w2L1_n16_seed0
  2. 測定  ``cmoe run --bench``      → result_logs/bench_slimpajama_depth_seed0

段2 は対照（幅2・深さ0 の配分）と候補（深さ1 の配分）を**同じ実行の中で**測る。
対応のある比較を実行をまたがずに取るためで、report/07 のベンチは測り直さない。

  uv run python experiments/15_search_depth/run.py --smoke   # 2層・少量
  uv run python experiments/15_search_depth/run.py --seed 0  # 見積り5時間
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
LOOKAHEAD = 1
WIDTH = 2
CALIB = 'slimpajama'
NSAMPLES = 16
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
NEXPERTS = 8
NACTIVE = 6
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 対照。report/07 の幅2・深さ0（同じ校正・同じ seed）の探索結果をそのまま読む
BASELINE_SEARCH = 'result_logs/slimpajama_w{width}_n{n}_seed{seed}'
# report/04 が測った dense。取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

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


def baseline_allocation(seed, plan):
    """対照（幅2・深さ0）の配分。measure し直さず、report/07 の探索結果を読む。

    smoke では対照が無いので、その場で幅2・深さ0 を走らせるのではなく一様を置く
    （smoke で見たいのは配線であって、比較ではない）。
    """
    if plan['layers']:
        return None
    path = ROOT / BASELINE_SEARCH.format(width=WIDTH, n=NSAMPLES, seed=seed)
    if not (path / 'search.json').exists():
        raise SystemExit(f'{path} が無い。対照になる幅{WIDTH}・深さ0 の探索が要る')
    record = load(path / 'search.json')
    if record['arguments']['calib'] != CALIB or record['search']['width'] != WIDTH:
        raise SystemExit(f'{path} は幅{WIDTH}・{CALIB} の探索ではない')
    return record['allocation']['values'], record['score']


def search_stage(seed, root, plan):
    """段1。幅は据え置き、深さだけ1段増やして探す。"""
    out_dir = root / f'{CALIB}_w{WIDTH}L{LOOKAHEAD}_n{plan["nsamples"]}_seed{seed}'
    argv = ['search', '--search', 'beam', '--width', str(WIDTH),
            '--lookahead', str(LOOKAHEAD), '--oracle', ORACLE,
            '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE)]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and record.get('seconds') is not None

    record = ensure(out_dir, 'search.json', argv, is_finished)
    allocation = record['allocation']
    log(f'  深さ{LOOKAHEAD}: score {record["score"]:.6e} '
        f'平均 x={allocation["mean_x"]:.3f} コスト {record["spent"]:g} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分)')
    log(f'  {allocation["values"]}')
    return {'stage': 'search', 'lookahead': LOOKAHEAD, 'width': WIDTH,
            'allocation': allocation['values'], 'score': record['score'],
            'spent': record['spent'], 'calls': record['calls'],
            'seconds': record['seconds'],
            'dir': str(out_dir.relative_to(ROOT))}


def bench_stage(candidate, baseline, seed, root, plan):
    """段2。対照と候補を同じ実行の中で測る。先頭が対応のある比較の基準になる。"""
    out_dir = root / f'bench_{CALIB}_depth_seed{seed}'
    argv = ['run', '--router', ROUTER, '--seeds', str(seed),
            '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--datasets', DATASETS,
            '--bench', '--bench-batch-size', str(plan['bench_batch'])]
    for spec in (baseline, candidate):
        argv += ['--alloc', spec]
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
                and len(record.get('runs', [])) == 2)

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'stage': 'bench', 'baseline': baseline, 'candidate': candidate,
            'dir': str(out_dir.relative_to(ROOT))}


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
    parser.add_argument('--no-bench', action='store_true',
                        help='段1（探索）だけで止める')
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='3層・少量。経路の確認')
    args = parser.parse_args(argv)

    plan = {
        'layers': 2 if args.smoke else None,
        # slimpajama は7成分に1本ずつ配るので、これ未満では引けない
        'nsamples': 7 if args.smoke else NSAMPLES,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    root = Path(args.out) if args.out else (
        ROOT / 'result_logs' / ('exp15_smoke' if args.smoke else ''))
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 {CALIB} n={plan["nsamples"]} / 幅 {WIDTH} 深さ {LOOKAHEAD}'
        + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'nsamples': plan['nsamples'],
              'width': WIDTH, 'lookahead': LOOKAHEAD, 'nexperts': NEXPERTS,
              'nactive': NACTIVE, 'stages': [], 'smoke': args.smoke}

    log(f'\n=== 段1 探索 幅{WIDTH}・深さ{LOOKAHEAD} / seed {args.seed} ===')
    stage = search_stage(args.seed, root, plan)
    record['stages'].append(stage)
    candidate = ','.join(str(x) for x in stage['allocation'])

    if not args.no_bench:
        control = baseline_allocation(args.seed, plan)
        if control is None:
            baseline = 'uniform4'
            log('\n=== 段2 測定（smoke。対照は一様） ===')
        else:
            values, score = control
            baseline = ','.join(str(x) for x in values)
            log(f'\n=== 段2 測定 / seed {args.seed} ===')
            log(f'  対照 幅{WIDTH}・深さ0: score {score:.6e}')
        record['stages'].append(
            bench_stage(candidate, baseline, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp15_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
