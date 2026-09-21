"""32 校正を n=64 に増やしたとき、ビーム幅は効くのか。

[report/30](../../report/30_beam-width-20seeds.md) は n=16 で幅1〜4 を測り、探索の
score は幅について単調に下がるのに PPL では単調にならなかった（幅1 対 幅2 は
seed 20本で差が標準誤差を下回る）。[report/31](../../report/31_calibration-amount.md)
は校正を n=16→64 に増やすと「探索 − 一様」の差が開くことを示した。**幅の効きを
n=64 で測り直す。**

  H1  n=64 では幅1 < 幅2 < 幅3 が PPL でも順に良くなる
  H0  n=16 と同じく、PPL では幅の順に並ばない

**動かすのは幅だけ**で、校正・動作点・ルーター・評価は report/31 の n=64 と同一
である。幅2 は report/31 で同じ機械・同じ校正トークンで済んでいるので読む。

段は2つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅1・幅3  → result_logs/slimpajama_w{1,3}_n64_seed<N>
  2. PPL   ``cmoe run``              → result_logs/ppl32_n64_seed<N>
     uniform3・幅1・幅2・幅3 を**同じ実行の中で**測る

段2 の先頭の uniform3 は `ppl31_n64_seed<N>` との**再現の検査**で、これが Δ=0 に
ならないうちは run をまたぐ比較を読んではいけない（report/22・26 と同じ理由）。

**3つの幅は同じ校正トークンを読む。** n も seed も同じなので当然だが、探索の
score を幅どうしで比べるにはこれが要る（report/31 の n をまたぐ比較では逆に、
ハッシュが相異なることを確かめていた）。段1 はハッシュの一致で止まる。

  uv run python experiments/32_width_at_n64/run.py --smoke   # 2層・n=7
  uv run python experiments/32_width_at_n64/run.py --seed 5  # 約4.6時間
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

# 動かすのはここだけ。校正も動作点も report/31 の n=64 と同じ
WIDTHS = (1, 2, 3)
NSAMPLES = 64
CALIB = 'slimpajama'
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
NEXPERTS = 8
NACTIVE = 6
DATASETS = 'wikitext2,c4-new'
# CLI へは1本の文字列で渡すが、再現の検査は1つずつ回る。**文字列をそのまま
# for で回すと1文字ずつ舐める**ので、名前の列はここで分けて持つ
DATASET_NAMES = tuple(DATASETS.split(','))

# report/31 の n=64 の PPL。段2 の uniform3 はこれと Δ=0 になるべきである
REPRODUCE = 'ppl31_n{n}_seed{seed}'

# n=64 では系列を分けて前向きする。z と隠れ状態がホスト側に残り、カードの峰が
# 47.3 → 44.0 GiB に下がる。値は変わらない（report/31「途中で起きたこと」）
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


def search_stage(seed, width, n_samples, layers, root):
    """幅1本の探索。幅2 は report/31 が同じ機械で済ませたものを読む。"""
    out_dir = root / f'{CALIB}_w{width}_n{n_samples}_seed{seed}'
    argv = ['search', '--calib', CALIB, '--nsamples', str(n_samples),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--search', 'beam', '--width', str(width)]
    if n_samples >= BATCH_CHUNK_FROM:
        argv += ['--batch-chunk', str(BATCH_CHUNK)]
    if layers:
        argv += ['--layers', str(layers)]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    log(f'  幅{width}: score {record["score"]:.6e} '
        f'平均 x={record["allocation"]["mean_x"]:.3f} '
        f'hash={record["calibration"]["token_hash"][:12]} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分)')
    return record, out_dir


def ppl_stage(seed, n_samples, layers, root, specs):
    out_dir = root / f'ppl32_n{n_samples}_seed{seed}'
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

    return ensure(out_dir, 'summary.json', argv, is_finished), out_dir


def check_reproduction(seed, n_samples, root, record, uniform):
    """先頭の uniform3 が report/31 の同じ構成と一致するか。

    run をまたいだ比較を読んでよいかの門である。Δ が 0 でないうちは、幅どうしの
    差が探索の違いなのか実行環境の違いなのかを分けられない。
    """
    path = LOGS / REPRODUCE.format(n=n_samples, seed=seed) / 'summary.json'
    if not path.exists():
        log(f'  再現の検査: {path.name} が無いので飛ばす')
        return
    before = load(path)
    old = [run for run in before['runs']
           if run['allocation']['name'] == uniform]
    new = [run for run in record['runs']
           if run['allocation']['name'] == uniform]
    if len(old) != 1 or len(new) != 1:
        raise SystemExit(f'{uniform} が report/31 側 {len(old)} 本、'
                         f'こちら {len(new)} 本')
    for name in DATASET_NAMES:
        gap = (new[0]['ppl']['cmoe'][name]['ppl']
               - old[0]['ppl']['cmoe'][name]['ppl'])
        log(f'  再現の検査 {name}: Δ={gap:+.3e}')
        if gap:
            raise SystemExit(
                f'{uniform} の {name} が report/31 と {gap:+.3e} 違う。'
                'run をまたぐ比較は読めない')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・n=7。配線の確認')
    parser.add_argument('--search-only', action='store_true',
                        help='段1 だけ。配分を出して止める')
    args = parser.parse_args(argv)

    layers = 2 if args.smoke else None
    # slimpajama は7成分に1本ずつ配るので 7 が下限
    n_samples = 7 if args.smoke else NSAMPLES
    root = LOGS / 'exp32_smoke' if args.smoke else LOGS
    root.mkdir(parents=True, exist_ok=True)
    log(f'seed {args.seed} / n={n_samples} / N={NEXPERTS} A={NACTIVE} / '
        f'幅 {", ".join(str(w) for w in WIDTHS)}'
        + ('（smoke）' if args.smoke else ''))

    started = time.time()
    searches, vectors, hashes, scores = {}, {}, {}, {}

    log('\n=== 段1 探索 ===')
    for width in WIDTHS:
        record, out_dir = search_stage(args.seed, width, n_samples, layers, root)
        searches[width] = str(out_dir.relative_to(ROOT))
        vectors[width] = record['allocation']['values']
        hashes[width] = record['calibration']['token_hash']
        scores[width] = record['score']

    # 幅どうしは同じ校正トークンを読む。違えば score を比べられない
    if len(set(hashes.values())) != 1:
        raise SystemExit(f'幅ごとに校正トークンが違う: '
                         + ', '.join(f'幅{w}={h[:12]}' for w, h in hashes.items()))
    log(f'  校正トークンは3つの幅で一致（{next(iter(hashes.values()))[:12]}）')

    stages = {'seed': args.seed, 'nsamples': n_samples, 'smoke': args.smoke,
              'widths': list(WIDTHS), 'search_dirs': searches,
              'vectors': {str(w): v for w, v in vectors.items()},
              'token_hash': next(iter(hashes.values())),
              'scores': {str(w): s for w, s in scores.items()}}

    uniform = f'uniform{NACTIVE // 2}'
    if args.smoke:
        # 2層の接頭辞は run --alloc に渡せない（全層ぶんを要求する）
        log('\n（smoke は探索まで。PPL の段は全層の配分が要る）')
    elif args.search_only:
        log('\n（--search-only。PPL の段は飛ばした）')
    else:
        log('\n=== 段2 PPL ===')
        specs = [uniform] + [','.join(str(x) for x in vectors[w]) for w in WIDTHS]
        record, out_dir = ppl_stage(args.seed, n_samples, layers, root, specs)
        check_reproduction(args.seed, n_samples, root, record, uniform)
        stages['uniform'] = uniform
        stages['ppl_dir'] = str(out_dir.relative_to(ROOT))

    stages.update({
        'seconds': time.time() - started,
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    })
    path = root / f'exp32_stages_seed{args.seed}.json'
    with path.open('w') as handle:
        json.dump(stages, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた / 所要 {stages["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
