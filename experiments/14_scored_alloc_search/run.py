"""14 答え部分で配分を探す。

report/11（experiments/09）は同じことを benchtrain でやった — 校正データの中身を
ベンチの train split に替え、そこで層ごとの x を探した。ここが替えるのは
**採点する位置**である。

  * 校正は `benchqa`（1問1系列、採点位置に印）
  * 分割を決める活性は採点位置だけで数える（`--profile-positions scored`）
  * **探索の目的関数の半分を答え部分に割り当てる**（`--scored-weight 0.5`）

3つ目がこの実験の新しいところで、experiments/13 に無かったものである。13 は
配分を `uniform4` に固定して分割の作り方だけを動かし、`acc` は +0.0015
[-0.0030, +0.0058] で有意にならなかった。動かしていなかったのは配分で、
探索の目的関数は「全位置の KL」のままだった（benchqa では位置の81%が埋めなので、
その目的関数は答え部分をほとんど見ていない）。

  H1  答え部分で探した配分が、同じ校正の一様配分を上回る → 位置の線は生きる
  H0  上回らない → 配分の探索でも位置は効かない。docs/04 へ資源を移す

**動作点は report/04・11・13 と同じ。** N=8 / A=6（スパース率25%）、
オラクル `suffix_kl`、ルーター `cmoe`、seed は 0 から。

**対照は同じ目的関数で選ぶ。** 一様 x=4,5,6 を、探索とまったく同じオラクル・
同じ重みで採点する（段2）。評価指標を見て対照を選ばない、という report/04 以来の
約束をここでも守る。

**重みは半分にする。** 答え部分の群が目的関数の 0.5、文脈の群が 0.5 を持つ
（埋めは 0）。位置あたりでは、答えの1位置が文脈の1位置の約3.2倍になる。全部を
答え部分に寄せる（s=1）のではなく半分から始めるのは、文脈の位置も「その問題を
読んでいる最中の活性」であって、捨てる理由がまだ無いからである。

**量は総トークンで揃える。** `--nsamples 1 --seqlen 1100` は採点位置 1,157・
文脈 3,720・埋め 11,337 の**総 16,214 トークン**で、report/11 の benchtrain
16,384 トークンと同じ量をモデルに通す。1問1系列では総トークンの7割が埋めなので、
これは「同じ計算量で測る」であって「同じ数の位置を重み付ける」ではない。重みを
持つのは 4,877 位置である。

`--seqlen` は PPL 評価の窓幅でもある。段1・2（探索と採点）は窓を切らないので
影響を受けないが、段3 まで進めたときの PPL は report/11 の 2048 窓と比べては
ならない（ベンチの acc は窓に依らないので比べられる）。

段は3つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅2 / 3  → result_logs/benchqa_s05_w<W>_seed<N>
  2. 採点  ``cmoe score``  一様 x=2..6 → result_logs/score_uniform_benchqa_s05_x2to6_seed<N>
  3. 測定  ``cmoe run --bench``（--with-bench を渡したときだけ）

  uv run python experiments/14_scored_alloc_search/run.py --smoke   # 2層・少量
  uv run python experiments/14_scored_alloc_search/run.py --seed 0  # 見積り1時間
  uv run python experiments/14_scored_alloc_search/run.py --seed 0 --with-bench

見積りは report/11 の同じ幅の探索（幅2 が 1224 秒、幅3 が 1833 秒、同じ総トークン
数）から取った。1問1系列は系列が短いぶん attention が安いので、これより速い。

**見ておくところ。** 1問1系列だと系列数が千の桁になる。捕捉（層0 の入力）は
一度に作られ、``--batch-chunk`` はそのあとの層まわしにしか効かない。読み出しは
重みが 0 でない位置しか持たない。この量なら小さいが、採点位置を揃える版
（`--nsamples 8`、総 364,844 位置）では、絞らないと読み出しだけで 23 GB になり
カードに載らない。
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

# この実験の3つの変更点（校正・数える位置・採点する位置）
CALIB = 'benchqa'
PROFILE_POSITIONS = 'scored'
# 答え部分が目的関数の半分を持つ
SCORED_WEIGHT = '0.5'
# 量は総トークンで揃える。1 × 1100 = 採点位置 1,157 で、総 16,214 トークンになる
NSAMPLES = 1
SEQLEN = 1100
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
WIDTHS = (2, 3)
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 動作点は report/04・11・13 と同じ。スパース率 25%
NEXPERTS = 8
NACTIVE = 6
# 対照。答え部分に重みを寄せた探索は平均 x=2.7 前後を出すので、量として釣り合う
# 一様は x=3 である（benchtrain の探索は 4.2 前後で、そちらは x=4 だった）
BASELINE = 'uniform3'
SCORED_UNIFORMS = ['uniform2', 'uniform3', 'uniform4', 'uniform5', 'uniform6']
# 段3 で測る一様。BASELINE のほかに、オラクルが最良と言った一様を並べる
# （評価指標ではなく探索と同じ目的関数で選んでいる）
BENCH_UNIFORMS = ['uniform2']
# 1問1系列だと系列が千の桁になるので、層まわしはバッチを分けて進める
BATCH_CHUNK = 128
# report/04 が測った dense。段3 で取り込む
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


def oracle_arguments(plan):
    """探索と採点で1文字も違ってはいけない引数。だから1箇所から生やす。"""
    argv = ['--oracle', ORACLE, '--calib', CALIB,
            '--nsamples', str(plan['nsamples']),
            '--seqlen', str(plan['seqlen']),
            '--profile-positions', PROFILE_POSITIONS,
            '--scored-weight', SCORED_WEIGHT,
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--batch-chunk', str(BATCH_CHUNK)]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv


def search_stage(width, seed, root, plan):
    """段1。幅 ``width`` のビームで、層ごとの x を探す。"""
    out_dir = root / f'benchqa_s05_w{width}_seed{seed}'
    argv = ['search', '--search', 'beam', '--width', str(width),
            '--seed', str(seed), *oracle_arguments(plan),
            '--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and record.get('seconds') is not None

    record = ensure(out_dir, 'search.json', argv, is_finished)
    allocation = record['allocation']
    log(f'  幅{width}: score {record["score"]:.6e} '
        f'平均 x={allocation["mean_x"]:.3f} {allocation["values"]}')
    return {'stage': 'search', 'width': width,
            'allocation': allocation['values'], 'score': record['score'],
            'dir': str(out_dir.relative_to(ROOT))}


def score_stage(seed, root, plan):
    """段2。一様配分を、探索とまったく同じオラクルで採点する。"""
    out_dir = root / f'score_uniform_benchqa_s05_x2to6_seed{seed}'
    argv = ['score', '--seed', str(seed), *oracle_arguments(plan)]
    for name in SCORED_UNIFORMS:
        argv += ['--alloc', name]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return len(record.get('scores', [])) == len(SCORED_UNIFORMS)

    record = ensure(out_dir, 'score.json', argv, is_finished)
    for row in record['scores']:
        log(f'  {row["spec"]}: score {row["score"]:.6e}')
    return {'stage': 'score',
            'scores': [{'spec': row['spec'], 'score': row['score']}
                       for row in record['scores']],
            'best': record['best'], 'dir': str(out_dir.relative_to(ROOT))}


def bench_stage(allocations, seed, root, plan):
    """段3。探索が出した配分と一様を、PPL と5タスクのベンチで測る。"""
    out_dir = root / f'bench_benchqa_s05_seed{seed}'
    argv = ['run', '--router', ROUTER, '--seeds', str(seed),
            '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--seqlen', str(plan['seqlen']),
            '--profile-positions', PROFILE_POSITIONS,
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--batch-chunk', str(BATCH_CHUNK), '--datasets', DATASETS,
            '--bench', '--bench-batch-size', str(plan['bench_batch'])]
    # 先頭が対応のある比較の基準になる。量として釣り合う一様を先に置き、
    # そのあとにオラクルが最良と言った一様を並べる
    for spec in [BASELINE, *BENCH_UNIFORMS, *allocations]:
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
                and 'summary' in record and 'bench_summary' in record)

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'stage': 'bench', 'dir': str(out_dir.relative_to(ROOT))}


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
    parser.add_argument('--widths', default=None,
                        help=f'走らせるビーム幅（既定は {",".join(map(str, WIDTHS))}）')
    parser.add_argument('--with-bench', action='store_true',
                        help='段3（変換して測る）まで進める')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・少量。経路の確認')
    args = parser.parse_args(argv)

    widths = ([int(field) for field in args.widths.split(',') if field.strip()]
              if args.widths else list(WIDTHS))

    plan = {
        'layers': 2 if args.smoke else None,
        'nsamples': NSAMPLES,
        'seqlen': 300 if args.smoke else SEQLEN,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke は問題を8問に絞るので、全件で測った dense は取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp14_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 {CALIB} 採点位置 {plan["nsamples"] * plan["seqlen"]} / '
        f'答え部分の重み {SCORED_WEIGHT} / '
        f'幅 {", ".join(map(str, widths))}' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB,
              'profile_positions': PROFILE_POSITIONS,
              'scored_weight': float(SCORED_WEIGHT),
              'nsamples': plan['nsamples'], 'seqlen': plan['seqlen'],
              'nexperts': NEXPERTS,
              'nactive': NACTIVE, 'widths': widths, 'stages': [],
              'smoke': args.smoke}

    allocations = []
    for width in widths:
        log(f'\n=== 段1 探索 幅{width} / seed {args.seed} ===')
        stage = search_stage(width, args.seed, root, plan)
        record['stages'].append(stage)
        allocations.append(','.join(str(x) for x in stage['allocation']))

    log(f'\n=== 段2 一様の採点 / seed {args.seed} ===')
    record['stages'].append(score_stage(args.seed, root, plan))

    if args.with_bench:
        log(f'\n=== 段3 測定 / seed {args.seed} ===')
        record['stages'].append(
            bench_stage(allocations, args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp14_stages_seed{args.seed}.json', record)
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
