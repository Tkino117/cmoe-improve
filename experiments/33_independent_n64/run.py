"""33 層間の依存を使わない配分（independent）を n=64 で測る。

[report/29](../../report/29_independent-kl.md) は n=16 で、層を前から順に決める
幅1 と、層を独立に決める independent を比べた。[report/31](../../report/31_calibration-amount.md)
で校正を n=64 に増やすと配分探索の効きが開くことが分かり、実験32 で幅1〜3 を
n=64 の 5 seed で測った。**その対照として independent を同じ条件で置く。**

採点オラクルは両方とも `suffix_kl`（dense モデルとの出力分布の KL）で、評価回数も
同じ 224 回である。違うのは、層を決めたあと次の層へ渡す状態だけである。

| 探索 | 次の層へ渡す状態 | 層 ℓ の候補を測るモデル |
|---|---|---|
| 幅1（実験32） | 決めた x で変換した状態 | 層 0..ℓ−1 を決めた x で変換、層 ℓ を候補で変換、層 ℓ+1.. は dense |
| **independent** | dense のまま | 層 ℓ だけを候補で変換、ほかの層は dense |

  H1  n=64 でも independent は幅1 に PPL で負ける（層間の依存を使う意味がある）
  H0  負けない

**校正・動作点・ルーター・評価は実験32 と同一**で、動かしたのは探索の種類だけ。
independent の層ごとのスコアは単独変換のモデルの値なので、`search.json` の
`score` と `recheck.score` は一致する。

段は2つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search --search independent`` → result_logs/slimpajama_indep_n64_seed<N>
  2. PPL   ``cmoe run``                        → result_logs/ppl33_n64_seed<N>
     uniform3 と independent を**同じ実行の中で**測る

段2 の先頭の uniform3 は `ppl32_n64_seed<N>` との**再現の検査**で、これが Δ=0 に
ならないうちは実験32 の幅1〜3 と並べて読んではいけない。

**校正トークンは実験32 と一致するはずである**（n も seed も同じ）。段1 はそれを
実験32 の幅1 の記録と突き合わせて確かめ、違えば止まる。

  uv run python experiments/33_independent_n64/run.py --smoke   # 2層・n=7
  uv run python experiments/33_independent_n64/run.py --seed 5  # 約1.3時間
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

# 動かすのはここだけ。校正も動作点も実験32 と同じ
SEARCH = 'independent'
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

# 実験32 の n=64 の PPL。段2 の uniform3 はこれと Δ=0 になるべきである
REPRODUCE = 'ppl32_n{n}_seed{seed}'
# 校正トークンの突き合わせ先。同じ n・同じ seed なので一致するはずである
CALIB_REFERENCE = 'slimpajama_w1_n{n}_seed{seed}'

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


def search_stage(seed, n_samples, layers, root):
    """層を独立に決める探索1本。評価回数は幅1 と同じ 224 回である。"""
    out_dir = root / f'{CALIB}_indep_n{n_samples}_seed{seed}'
    argv = ['search', '--calib', CALIB, '--nsamples', str(n_samples),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--search', SEARCH]
    if n_samples >= BATCH_CHUNK_FROM:
        argv += ['--batch-chunk', str(BATCH_CHUNK)]
    if layers:
        argv += ['--layers', str(layers)]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    log(f'  independent: score {record["score"]:.6e} '
        f'平均 x={record["allocation"]["mean_x"]:.3f} '
        f'hash={record["calibration"]["token_hash"][:12]} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分)')
    # 層ごとのスコアが単独変換のモデルの値なので、頭から測り直しても同じになる
    gap = abs(record['recheck']['score'] - record['score'])
    if gap:
        raise SystemExit(f'independent は測り直しで一致するはずだが {gap:.3e} 違う')
    return record, out_dir


def check_calibration(seed, n_samples, token_hash):
    """実験32 の幅1 と同じ校正トークンを読んでいるか。

    n も seed も同じなので一致するはずで、違えば「同じ条件の対照」ではない。
    実験32 が無い seed では飛ばす。
    """
    path = LOGS / CALIB_REFERENCE.format(n=n_samples, seed=seed) / 'search.json'
    if not path.exists():
        log(f'  校正の突き合わせ: {path.parent.name} が無いので飛ばす')
        return
    other = load(path)['calibration']['token_hash']
    if other != token_hash:
        raise SystemExit(
            f'校正トークンが実験32 の幅1 と違う（{token_hash[:12]} 対 {other[:12]}）')
    log(f'  校正トークンは実験32 の幅1 と一致（{token_hash[:12]}）')


def ppl_stage(seed, n_samples, layers, root, specs):
    out_dir = root / f'ppl33_n{n_samples}_seed{seed}'
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
    """先頭の uniform3 が実験32 の同じ構成と一致するか。

    run をまたいだ比較を読んでよいかの門である。Δ が 0 でないうちは、
    independent と幅1〜3 の差が探索の違いなのか実行環境の違いなのかを分けられない。
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
        raise SystemExit(f'{uniform} が実験32 側 {len(old)} 本、'
                         f'こちら {len(new)} 本')
    for name in DATASET_NAMES:
        gap = (new[0]['ppl']['cmoe'][name]['ppl']
               - old[0]['ppl']['cmoe'][name]['ppl'])
        log(f'  再現の検査 {name}: Δ={gap:+.3e}')
        if gap:
            raise SystemExit(
                f'{uniform} の {name} が実験32 と {gap:+.3e} 違う。'
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
    root = LOGS / 'exp33_smoke' if args.smoke else LOGS
    root.mkdir(parents=True, exist_ok=True)
    log(f'seed {args.seed} / n={n_samples} / N={NEXPERTS} A={NACTIVE} / '
        f'探索 {SEARCH}' + ('（smoke）' if args.smoke else ''))

    started = time.time()

    log('\n=== 段1 探索 ===')
    record, out_dir = search_stage(args.seed, n_samples, layers, root)
    vector = record['allocation']['values']
    token_hash = record['calibration']['token_hash']
    check_calibration(args.seed, n_samples, token_hash)

    stages = {'seed': args.seed, 'nsamples': n_samples, 'smoke': args.smoke,
              'search': SEARCH,
              'search_dir': str(out_dir.relative_to(ROOT)),
              'vector': vector, 'token_hash': token_hash,
              'score': record['score']}

    uniform = f'uniform{NACTIVE // 2}'
    if args.smoke:
        # 2層の接頭辞は run --alloc に渡せない（全層ぶんを要求する）
        log('\n（smoke は探索まで。PPL の段は全層の配分が要る）')
    elif args.search_only:
        log('\n（--search-only。PPL の段は飛ばした）')
    else:
        log('\n=== 段2 PPL ===')
        specs = [uniform, ','.join(str(x) for x in vector)]
        summary, ppl_dir = ppl_stage(args.seed, n_samples, layers, root, specs)
        check_reproduction(args.seed, n_samples, root, summary, uniform)
        stages['uniform'] = uniform
        stages['ppl_dir'] = str(ppl_dir.relative_to(ROOT))

    stages.update({
        'seconds': time.time() - started,
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    })
    path = root / f'exp33_stages_seed{args.seed}.json'
    with path.open('w') as handle:
        json.dump(stages, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた / 所要 {stages["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
