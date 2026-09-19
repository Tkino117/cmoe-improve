"""31 校正データの量（n=16/32/64）に対する配分探索のロバスト性を測る。

論文は「探索が参照するのは校正データのみ」を売りにしており、**その校正がどれだけ
要るのか**に答えが無い。report/25 で並べた静的枝刈りの対照は、原典では FLAP が
262k トークン・LLM-Pruner が 640 トークンで、提案手法の 32k は両者の中間にある。
査読でここは突かれる。

  H1  n を 16→32→64 と増やしても、探索が選ぶ配分と PPL はほとんど動かない
  H0  動く（32k では足りていない）

**n は探索と分割の両方に効かせる**（`cmoe run --nsamples` が両者を1セットで
共有している、リポジトリの現行の取り決めそのまま）。したがって測っているのは
「手法全体の校正予算」であって、探索だけの感度ではない。同じ n の一様配分を
対照に置くのはこのためで、**分割が変わったぶんは一様の側にも同じだけ乗る**。

段は2つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅2  → result_logs/slimpajama_w2_n{16,32,64}_seed<N>
  2. PPL   ``cmoe run``         → result_logs/ppl31_n{16,32,64}_seed<N>
     uniform3 と探した配分を、その n の校正で組んだ上で同じ実行の中で測る

**seed は 5 / 6 / 7 を使う。** n=16 幅2 の探索はこの seed ではこの機械で済んで
おり（33.4〜33.7分、report/30 の 33.7±0.2 と一致）、段1 はそれを読む。seed 0〜2 の
n=16 は別マシン（report/07）なので混ぜない。

交絡が1つある。slimpajama は7成分に最低1本ずつ配ってから残りを比率で配るので、
n を上げると成分比が真の比率へ寄る（CommonCrawl は 37.5% → 43.8% → 50.0%、
実比率 54.45%）。「量が増えた」と「混合物が本来の比率に近づいた」は分かれない。

  uv run python experiments/31_calib_amount/run.py --smoke   # 2層・n=7/14
  uv run python experiments/31_calib_amount/run.py --seed 5  # 約3.4時間
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
LOGS = ROOT / 'result_logs'

# 動かすのはここだけ。幅も動作点もルーターも report/30 の主系と同じ
NSAMPLES = (16, 32, 64)
WIDTH = 2
CALIB = 'slimpajama'
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
NEXPERTS = 8
NACTIVE = 6
DATASETS = 'wikitext2,c4-new'

# smoke の n。slimpajama は7成分に1本ずつ配るので 7 が下限
SMOKE_NSAMPLES = (7, 14)

# n がこれ以上なら系列を分けて前向きする。z と隠れ状態がホスト側に残り、カードの
# 峰が n=64 の2層探索で 47.3 → 44.0 GiB（全 48.9）に下がる。**値は変わらない**
# （同じ2層探索で score が 4.582743e-03 で一致、差 0.000e+00）。n=16 の既存の
# 記録とも食い違わない — batch_chunk は NOT_IDENTITY に入れてある
BATCH_CHUNK_FROM = 32
BATCH_CHUNK = 16

# 結果を変えない引数。同じ実験かの判定から外す
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


def differences(record, argv):
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
            if record is not None and differences(record, argv):
                raise SystemExit(
                    f'{out_dir} は違う実験の出力である'
                    f'（{", ".join(differences(record, argv))}）')
            retire(out_dir)
        run_cli(argv)
        record = load(payload) if payload.exists() else None
        if record is None or not is_finished(record):
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    found = differences(record, argv)
    if found:
        raise SystemExit(f'{out_dir} は違う実験の出力である（{", ".join(found)}）')
    return record


def search_stage(seed, n_samples, layers, root):
    """幅2 の探索1本。n=16 は report/30 が同じ機械で済ませたものを読む。"""
    out_dir = root / f'{CALIB}_w{WIDTH}_n{n_samples}_seed{seed}'
    argv = ['search', '--calib', CALIB, '--nsamples', str(n_samples),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--search', 'beam', '--width', str(WIDTH)]
    if n_samples >= BATCH_CHUNK_FROM:
        argv += ['--batch-chunk', str(BATCH_CHUNK)]
    if layers:
        argv += ['--layers', str(layers)]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    log(f'  n={n_samples}: score {record["score"]:.6e} '
        f'平均 x={record["allocation"]["mean_x"]:.3f} '
        f'hash={record["calibration"]["token_hash"][:12]} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分)')
    return record, out_dir


def ppl_stage(seed, n_samples, layers, root, specs):
    """その n の校正で組んで PPL を測る。一様も同じ実行の中で組み直す。"""
    out_dir = root / f'ppl31_n{n_samples}_seed{seed}'
    specs = list(dict.fromkeys(specs))
    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(n_samples),
             '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
             '--datasets', DATASETS]
    if n_samples >= BATCH_CHUNK_FROM:
        argv += ['--batch-chunk', str(BATCH_CHUNK)]
    if layers:
        argv += ['--layers', str(layers)]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return (not record.get('failures') and 'summary' in record
                and len(record.get('runs', [])) == len(specs))

    record = ensure(out_dir, 'summary.json', argv, is_finished)
    return record, out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・n=7/14。配線の確認')
    parser.add_argument('--search-only', action='store_true',
                        help='段1 だけ。配分を出して止める')
    args = parser.parse_args(argv)

    layers = 2 if args.smoke else None
    n_list = SMOKE_NSAMPLES if args.smoke else NSAMPLES
    root = LOGS / 'exp31_smoke' if args.smoke else LOGS
    root.mkdir(parents=True, exist_ok=True)
    log(f'seed {args.seed} / 幅{WIDTH} / N={NEXPERTS} A={NACTIVE} / '
        f'n={", ".join(str(n) for n in n_list)}'
        + ('（smoke）' if args.smoke else ''))

    started = time.time()
    searches, vectors, hashes, scores = {}, {}, {}, {}

    log('\n=== 段1 探索 ===')
    for n_samples in n_list:
        record, out_dir = search_stage(args.seed, n_samples, layers, root)
        searches[n_samples] = str(out_dir.relative_to(ROOT))
        vectors[n_samples] = record['allocation']['values']
        hashes[n_samples] = record['calibration']['token_hash']
        scores[n_samples] = record['score']

    # n が違えば校正トークンも違う。同じなら n が効いていないか、引き方の誤り
    if len(set(hashes.values())) != len(hashes):
        raise SystemExit(f'n の違う校正が同じトークンになっている: {hashes}')

    stages = {'seed': args.seed, 'width': WIDTH, 'smoke': args.smoke,
              'n_samples': list(n_list), 'search_dirs': searches,
              'vectors': {str(n): v for n, v in vectors.items()},
              'token_hashes': {str(n): h for n, h in hashes.items()},
              'scores': {str(n): s for n, s in scores.items()}}

    if args.smoke:
        # 2層の接頭辞は run --alloc に渡せない（全層ぶんを要求する）
        log('\n（smoke は探索まで。PPL の段は全層の配分が要る）')
    elif args.search_only:
        log('\n（--search-only。PPL の段は飛ばした）')
    else:
        log('\n=== 段2 PPL ===')
        uniform = f'uniform{NACTIVE // 2}'
        ppl_dirs = {}
        for n_samples in n_list:
            spec = ','.join(str(x) for x in vectors[n_samples])
            _, out_dir = ppl_stage(args.seed, n_samples, layers, root,
                                   [uniform, spec])
            ppl_dirs[str(n_samples)] = str(out_dir.relative_to(ROOT))
        stages['uniform'] = uniform
        stages['ppl_dirs'] = ppl_dirs

    stages.update({
        'seconds': time.time() - started,
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    })
    path = root / f'exp31_stages_seed{args.seed}.json'
    with path.open('w') as handle:
        json.dump(stages, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた / 所要 {stages["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
