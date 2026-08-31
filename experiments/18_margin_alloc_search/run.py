"""18 採点そのもので配分を探す。

report/11 と report/14 が繰り返し出したのは、**探索の目的関数の順位が `acc` の
順位と一致しない**という結果だった。report/14 段2 では、beam3 が KL で一様配分に
勝ちながら（0.118864 対 0.136491）`acc` で 0.6113 対 0.6170 と負けている。校正
データの中身を評価に寄せ（report/11）、活性を数える位置を採点位置に寄せ
（report/13）、目的関数の重みを答え部分に寄せ（report/14）ても、目的関数そのものは
**親モデルとの KL** のままだった。

ここが替えるのは目的関数そのものである。

  * 校正は `benchchoice`（1問を**選択肢ごとに1系列**。不正解肢も入る）
  * 目的関数は `margin`（正解と、最も惜しい不正解の対数尤度の差）
  * 分割を決める活性は採点位置だけで数える（`--profile-positions scored`）

マージンの定義は

    M = -ll(正解) + (1/β) logsumexp( β · ll(不正解) )     小さいほど良い

で、β の両端に評価側の指標が立っている。β → ∞ で `bench_stats.margin` の符号
反転（その符号が `acc`）、β = 1 に softplus を掛けると `bench_stats.gold_nll` と
厳密に一致する。既定は β = 1・損失 raw で、その族の中の一点である。

  H1  マージンで探した配分が、同じ校正の一様配分を `acc` で上回る
      → 「目的関数が KL だったから効かなかった」が言える。docs/04 の柱になる
  H0  上回らない → 配分の探索は目的関数を替えても下流に効かない。
      探索ではなく校正データの側（docs/04）に資源を寄せる

**動作点は report/04・11・13・14 と同じ。** N=8 / A=6（スパース率25%）、
ルーター `cmoe`、seed は 0 から。

**量は問題数で揃える。** `--nsamples 5 --seqlen 25` で 125 問（5タスクへ25問ずつ）。
`benchqa` が採点位置で数えていたのをここで替えている — マージンは問題ごとの量
なので、位置で配ると問題数が続きの長さに反比例して決まってしまう（正解肢の採点
位置を等しく配ると、PIQA 11問に対して HellaSwag は7問にしかならない）。
125問・400系列・総 61,600 トークンで、`benchqa` の 16,214 トークンの 3.8 倍を
モデルに通す。**同じ計算量ではない** — 揃えているのは問題の数である。

**dense は下限ではない。** KL は dense がスコア 0 の下限を持っていたが、マージン
にそれは無く、探索が dense を下回ることは起こりうる（実際、2層のスモークでは
dense −3.134 に対して −3.445 が出た）。それが狙いだが、同時に**125問への過適合と
区別が付かない**ということでもある。段2 の一様配分と、段3 の本番のベンチが
その区別を付ける唯一の材料である。

**対照は同じ目的関数で選ぶ。** 一様 x=2..6 を、探索とまったく同じオラクルで
採点する（段2）。評価指標を見て対照を選ばない、という report/04 以来の約束を
ここでも守る。

段は3つで、**済んだ段は飛ばす**。

  1. 探索  ``cmoe search`` 幅2 / 3 / 4 → result_logs/margin_w<W>_seed<N>
  2. 採点  ``cmoe score``  一様 x=2..6 → result_logs/score_uniform_margin_x2to6_seed<N>
  3. 測定  ``cmoe run --bench``（--with-bench を渡したときだけ）

  uv run python experiments/18_margin_alloc_search/run.py --smoke   # 2層・少量
  uv run python experiments/18_margin_alloc_search/run.py --seed 0  # 見積り3.7時間
  uv run python experiments/18_margin_alloc_search/run.py --seed 0 --with-bench

見積りは report/14 の同じ形の探索（幅2 が 773 秒、16,214 トークン）から、
トークン数に比例させて取った。幅3 は幅2 の約1.5倍、幅4 は約2倍である。

**見ておくところ。** `search.json` の層ごとの内訳に `calibration_acc` が入る
（マージンの符号なので、追加の forward 無しに出る）。探索が目的関数を下げながら
校正上の正答率を落としていないか、走っている最中に見られる。落としているなら、
β か損失の形を疑う。
"""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

from cmoe.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]

# この実験の変更点は目的関数そのもの
CALIB = 'benchchoice'
ORACLE = 'margin'
PROFILE_POSITIONS = 'scored'
# β=1 は softplus と組めば gold_nll と一致する点。損失は書いたとおりの raw
MARGIN_BETA = '1.0'
MARGIN_LOSS = 'raw'
# 量は問題数で揃える。5 × 25 = 125 問（5タスクへ25問ずつ）
NSAMPLES = 5
SEQLEN = 25
ROUTER = 'cmoe'
WIDTHS = (2, 3, 4)
BENCH_BATCH = 32
# ベンチが1系列に許す長さ。``--seqlen`` は校正の量（問題数）を決めるために 25 に
# なっているが、それは lm-eval が読める長さではない。2048 は Llama-2 の文脈長で、
# report/11・13・16 のベンチが通っていた値でもある（この5タスクでは1問も切れない）
BENCH_MAX_LENGTH = 2048
# 動作点は report/04・11・13・14 と同じ。スパース率 25%
NEXPERTS = 8
NACTIVE = 6
# report/11・13・14 が基準に使ってきた一様。対応のある比較の基準になる
BASELINE = 'uniform4'
SCORED_UNIFORMS = ['uniform2', 'uniform3', 'uniform4', 'uniform5', 'uniform6']
# 1問K系列だと系列が数百になるので、層まわしはバッチを分けて進める
BATCH_CHUNK = 128
# report/04 が測った dense。段3 で取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'

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
            '--margin-beta', MARGIN_BETA, '--margin-loss', MARGIN_LOSS,
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--batch-chunk', str(BATCH_CHUNK)]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    return argv


def search_stage(width, seed, root, plan):
    """段1。幅 ``width`` のビームで、層ごとの x を探す。"""
    out_dir = root / f'margin_w{width}_seed{seed}'
    argv = ['search', '--search', 'beam', '--width', str(width),
            '--seed', str(seed), *oracle_arguments(plan),
            '--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and record.get('seconds') is not None

    record = ensure(out_dir, 'search.json', argv, is_finished)
    allocation = record['allocation']
    dense = record.get('dense_score')
    log(f'  幅{width}: score {record["score"]:.6f} '
        f'（dense {dense:.6f}）平均 x={allocation["mean_x"]:.3f} '
        f'{allocation["values"]}')
    return {'stage': 'search', 'width': width,
            'allocation': allocation['values'], 'score': record['score'],
            'dense_score': dense, 'dir': str(out_dir.relative_to(ROOT))}


def score_stage(seed, root, plan):
    """段2。一様配分を、探索とまったく同じオラクルで採点する。"""
    out_dir = root / f'score_uniform_margin_x2to6_seed{seed}'
    argv = ['score', '--seed', str(seed), *oracle_arguments(plan)]
    for name in SCORED_UNIFORMS:
        argv += ['--alloc', name]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return len(record.get('scores', [])) == len(SCORED_UNIFORMS)

    record = ensure(out_dir, 'score.json', argv, is_finished)
    for row in record['scores']:
        log(f'  {row["spec"]}: score {row["score"]:.6f}')
    return {'stage': 'score',
            'scores': [{'spec': row['spec'], 'score': row['score']}
                       for row in record['scores']],
            'best': record['best'], 'dir': str(out_dir.relative_to(ROOT))}


def bench_stage(allocations, uniforms, seed, root, plan):
    """段3。探索が出した配分と一様を、5タスクのベンチで測る。

    **一様も測り直す。** 変換そのものが校正データに依存する（``benchchoice`` の
    400系列から expert を切り出す）ので、別の校正で切った既存の uniform は
    ここでは対照にならない。report/11 が出したのは「校正データを替えるだけで
    `acc` が +0.036 動く」で、これはこれまでに見たどの配分の効果より大きい。
    流用すると配分の効果と校正の効果が混ざる。**dense だけは校正に依存しない**
    ので、report/04 の測定を ``--bench-reference`` で取り込む。

    **PPL は測らない。** ``--seqlen`` は校正の量（``n_samples × seqlen`` = 問題数）
    と PPL の窓幅、そしてベンチが1系列に許す長さの3つを兼ねている。ここは 25 に固定されている — 変換を探索と同じに
    するには校正を1トークンも変えられないからで、25トークンの窓で測った PPL は
    どの既存の測定とも比べられない。仮説は `acc` の側にあるので、意味を持たない
    数を出すより測らない。ベンチの側は ``--bench-max-length`` で 2048 に戻す
    （そこを 25 のままにすると、選択肢が入らずに lm-eval が落ちる）。
    """
    out_dir = root / f'bench_margin_seed{seed}'
    argv = ['run', '--router', ROUTER, '--seeds', str(seed),
            '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--seqlen', str(plan['seqlen']),
            '--profile-positions', PROFILE_POSITIONS,
            '--nexperts', str(NEXPERTS), '--nactive', str(NACTIVE),
            '--batch-chunk', str(BATCH_CHUNK), '--no-ppl',
            '--bench', '--bench-batch-size', str(plan['bench_batch']),
            '--bench-max-length', str(BENCH_MAX_LENGTH)]
    # 先頭が対応のある比較の基準になる。report/11・13・14 と同じ一様を先に置き、
    # そのあとにオラクルが最良と言った一様を並べる
    extra = [name for name in uniforms if name != BASELINE]
    for spec in [BASELINE, *extra, *allocations]:
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
        return not record.get('failures') and 'bench_summary' in record

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
        'seqlen': 5 if args.smoke else SEQLEN,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        # smoke は問題を8問に絞るので、全件で測った dense は取り込めない
        'reference': None if args.smoke else DENSE_REFERENCE,
    }
    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp18_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    log(f'seed {args.seed} / N={NEXPERTS} A={NACTIVE}'
        f'（スパース率 {100 * (NEXPERTS - NACTIVE) // NEXPERTS}%）/ '
        f'校正 {CALIB} {plan["nsamples"] * plan["seqlen"]} 問 / '
        f'β={MARGIN_BETA} 損失 {MARGIN_LOSS} / '
        f'幅 {", ".join(map(str, widths))}' + ('（smoke）' if args.smoke else ''))
    log(f'出力 {root}')

    started = time.time()
    record = {'seed': args.seed, 'calib': CALIB, 'oracle': ORACLE,
              'profile_positions': PROFILE_POSITIONS,
              'margin_beta': float(MARGIN_BETA), 'margin_loss': MARGIN_LOSS,
              'nsamples': plan['nsamples'], 'seqlen': plan['seqlen'],
              'nexperts': NEXPERTS, 'nactive': NACTIVE, 'widths': widths,
              'stages': [], 'smoke': args.smoke}

    allocations = []
    for width in widths:
        log(f'\n=== 段1 探索 幅{width} / seed {args.seed} ===')
        stage = search_stage(width, args.seed, root, plan)
        record['stages'].append(stage)
        allocations.append(','.join(str(x) for x in stage['allocation']))

    log(f'\n=== 段2 一様配分を同じオラクルで採点 / seed {args.seed} ===')
    scored = score_stage(args.seed, root, plan)
    record['stages'].append(scored)

    if args.with_bench:
        log(f'\n=== 段3 変換して測る / seed {args.seed} ===')
        record['stages'].append(bench_stage(
            allocations, [scored['best']['spec']], args.seed, root, plan))

    record['seconds'] = time.time() - started
    summarize(root / f'exp18_stages_seed{args.seed}.json', record)
    log(f'{record["seconds"] / 60:.1f} 分')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
