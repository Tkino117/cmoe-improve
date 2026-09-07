"""17 既に測った構成を、MMLU でもう一度測る。

MMLU（test 14,042問）は今回ベンチに足したばかりで、slimpajama 校正の測定
（[report/07](../../report/07_slimpajama-alloc-3seeds.md) の25%、
[report/08](../../report/08_sparsity50-alloc-3seeds.md) の50%、
`experiments/16_sparsity75` の75%）にはどれも入っていない。

**測り直すのは MMLU だけである。** PPL と5タスクは既に測ってあり、同じ引数で
引き直しても同じ数字が出るだけなので、`--no-ppl --bench-tasks mmlu` で MMLU の
生の尤度だけを取りに行く。既存の `bench_slimpajama*` は1つも触らない。

**構成は既存の実行からそのまま引く。** 探索が出した配分は seed ごとに違うので、
ここで作り直すと別物になる。元の `bench_.../summary.json` の ``arguments.alloc``
を順番ごと写して渡す（先頭が対応のある比較の基準になるので、順番も意味を持つ）。

  25%  bench_slimpajama_seed<N>     10構成 → bench_mmlu_slimpajama_seed<N>
  50%  bench_slimpajama_a4_seed<N>   8構成 → bench_mmlu_slimpajama_a4_seed<N>

**元の実行は ``--set`` で選ぶ。** 既定の ``main`` は上の一様＋探索の表である。
``baselines`` は [report/21](../../report/21_ew-rule-baseline.md) の EW-rule と
[report/22](../../report/22_alloc-baselines.md) の対照5行で、こちらは5タスクと
PPL だけ測ってあって MMLU が無い。res16 / res22 の MMLU 列の「—」がこれで埋まる。

  baselines 25%  bench22_seed<N> + ew_bench_seed<N>       → bench_mmlu_baselines_seed<N>
  baselines 50%  bench22_a4_seed<N> + ew_bench_a4_seed<N> → bench_mmlu_baselines_a4_seed<N>

対照の一様（A=6 なら uniform3、A=4 なら uniform2）は両方の元に入っているので、
束ねるときに1本に畳む。**先頭に来るのはこの一様で、対応のある比較の基準になる。**
main 側で同じ配分の MMLU を既に測ってあり、突き合わせれば run をまたいで
同じ表に並べてよいことをその場で確かめられる（report/21 と同じ再現の検査）。

**75%（A=2）は測らない。** 実験16 で5タスクの acc が chance（macro 0.35）に
張り付くところまで壊れていることが分かった。MMLU の chance は 0.25 なので、
測っても床を確かめるだけになる。必要になったら ``--nactive 2`` で回せる。

dense の基準は既存のものを使えない（`bench_h4` の dense は5タスクぶんしか無く、
``cmoe run`` はタスクの顔ぶれが違えば取り込みを断る）。**最初の1回だけ測り**、
残りは全部それを取り込む。置き場所は ``main`` の A=6 / seed 0 の実行の中で、
``baselines`` も同じものを取り込む（dense は配分にも校正にも依らない）。

  uv run python experiments/17_mmlu_bench/run.py --smoke            # 2層・科目あたり2問
  uv run python experiments/17_mmlu_bench/run.py --nactive 6 --seed 0  # dense も測る
  uv run python experiments/17_mmlu_bench/run.py --nactive 6 --seed 1
  ...
  uv run python experiments/17_mmlu_bench/run.py --all              # 25% と 50% × 3 seed
  uv run python experiments/17_mmlu_bench/run.py --set baselines --all   # 対照の穴埋め

まとめは 06・07・16 の ``summarize_seeds.py`` とは別で、report を書くときに
``cmoe.eval.bench.load_samples()`` で5タスクぶんと束ねる。
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

CALIB = 'slimpajama'
NSAMPLES = 16
ROUTER = 'cmoe'
NEXPERTS = 8
# 測るのは MMLU だけ。5タスクは既存の実行にある
TASKS = 'mmlu'
BENCH_BATCH = 32
# --nactive に渡せる動作点。25% / 50% / 75%
NACTIVES = (6, 4, 2)
# ``--all`` が回す動作点。**75%（A=2）は入れない** — 実験16 で5タスクの acc が
# chance（macro 0.35）に張り付くところまで壊れており、MMLU の chance は 0.25
# なので、測っても床を確かめるだけになる。必要になったら
# ``--nactive 2 --seed <N>`` で1本ずつ回せる
ALL_NACTIVES = (6, 4)
SEEDS = (0, 1, 2)
# dense を測る1回。ここ以外は全部これを取り込む（dense は変換していないモデル
# なので、A にも校正にも seed にも配分にも依らない）
DENSE_AT = (6, 0)

# 元の実行の置き場所。``{tag}`` は A=6 で空、それ以外は ``_a<A>``。
# 束ねる順がそのまま ``--alloc`` の順になり、**先頭が対応のある比較の基準**に
# なるので、対照の一様を先頭に持つ側を先に並べる
SOURCE_SETS = {
    'main': ('bench_{calib}{tag}_seed{seed}',),
    'baselines': ('bench22{tag}_seed{seed}', 'ew_bench{tag}_seed{seed}'),
}
# 測り先。``main`` は report/17 の頃の名前を保つ
TARGET_SETS = {
    'main': 'bench_mmlu_{calib}{tag}_seed{seed}',
    'baselines': 'bench_mmlu_baselines{tag}_seed{seed}',
}

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


def tag(nactive):
    """動作点の札。A=6 だけは名前に入らない（report/06・07 の頃の名前を保つ）。"""
    return '' if nactive == 6 else f'_a{nactive}'


def source_dirs(root, source_set, nactive, seed):
    return [root / name.format(calib=CALIB, tag=tag(nactive), seed=seed)
            for name in SOURCE_SETS[source_set]]


def target_dir(root, source_set, nactive, seed):
    return root / TARGET_SETS[source_set].format(
        calib=CALIB, tag=tag(nactive), seed=seed)


def dense_reference(root):
    """dense は1つだけ。``main`` の A=6 / seed 0 の中に置いてある。"""
    return target_dir(root, 'main', *DENSE_AT) / 'bench' / 'dense.json'


def source_allocations(root, source_set, nactive, seed):
    """元の実行が測った配分を、順番ごと写して束ねる。

    探索や規則が出した配分は seed ごとに違うので、ここで作り直すと別物になる。
    先頭は対応のある比較の基準なので、順番まで元のとおりにする。元が2つ以上
    あるときは並べた順につなぎ、重なる配分（どちらにも入っている対照の一様）は
    先に出たほうを残して畳む。
    """
    specs = []
    for directory in source_dirs(root, source_set, nactive, seed):
        payload = directory / 'summary.json'
        if not payload.exists():
            raise SystemExit(
                f'{payload} が無い。MMLU は既に測った構成に足すものなので、'
                f'元の実行（A={nactive} / seed {seed}）を先に通すこと')
        record = load(payload)
        if 'bench_summary' not in record or record.get('failures'):
            raise SystemExit(f'{payload} は終わっていない実行である')
        if record['arguments']['nactive'] != nactive:
            raise SystemExit(
                f'{payload} は A={record["arguments"]["nactive"]} の実行である')
        specs.extend(record['arguments']['alloc'])
    return list(dict.fromkeys(specs))


def bench_stage(root, source_set, nactive, seed, plan):
    """1つの動作点・1つの seed を MMLU で測る。"""
    out_dir = target_dir(root, source_set, nactive, seed)
    specs = plan['allocations']
    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(nactive),
             # PPL は既存の実行にある。ここで引き直しても同じ数字が出るだけ
             '--no-ppl',
             '--bench', '--bench-tasks', TASKS,
             '--bench-batch-size', str(plan['bench_batch'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    if plan['reference'] is not None:
        argv += ['--bench-reference', str(plan['reference'])]
        log(f'  dense は {plan["reference"].relative_to(ROOT)} から取り込む')
    else:
        log('  dense をここで測る（この1回だけ。残りはこれを取り込む）')
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        # --no-ppl なので 'summary' は入らない。見るのはベンチ側だけ
        return (not record.get('failures')
                and 'bench_summary' in record
                and len(record.get('runs', [])) == len(specs))

    ensure(out_dir, 'summary.json', argv, is_finished)
    return {'dir': str(out_dir.relative_to(ROOT)), 'allocations': specs}


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


def jobs(args):
    """走らせる (A, seed) の並び。

    dense を測る1本を必ず先頭に置く。残りは ``NACTIVES`` の順（25% → 50%）で
    並べる。``--all`` に 75% は入らない（``ALL_NACTIVES`` を見ること）。
    """
    if args.all:
        wanted = [(nactive, seed) for nactive in ALL_NACTIVES for seed in SEEDS]
    else:
        wanted = [(args.nactive, args.seed)]
    return sorted(wanted,
                  key=lambda job: (job != DENSE_AT, NACTIVES.index(job[0]), job[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nactive', type=int, default=6, choices=NACTIVES,
                        help='動作点。6=25%% / 4=50%% / 2=75%%')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--set', dest='source_set', default='main',
                        choices=sorted(SOURCE_SETS),
                        help='元の実行。main=一様＋探索 / '
                             'baselines=EW-rule と report/22 の対照5行')
    parser.add_argument('--all', action='store_true',
                        help='25%% と 50%% を 3 seed ずつまとめて回す（75%% は入らない）')
    parser.add_argument('--out', default=None, help='出力の置き場所')
    parser.add_argument('--smoke', action='store_true',
                        help='2層・科目あたり2問。経路の確認')
    args = parser.parse_args(argv)

    default_root = ROOT / 'result_logs'
    if args.smoke:
        default_root = default_root / 'exp17_smoke'
    root = Path(args.out) if args.out else default_root
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record = {'calib': CALIB, 'tasks': TASKS, 'nexperts': NEXPERTS,
              'source_set': args.source_set, 'smoke': args.smoke, 'runs': []}

    for nactive, seed in jobs(args):
        log(f'\n=== MMLU / {args.source_set} / N={NEXPERTS} A={nactive}'
            f'（スパース率 {100 * (NEXPERTS - nactive) // NEXPERTS}%）'
            f' / seed {seed} ===')
        if args.smoke:
            sources = source_dirs(ROOT / 'result_logs', args.source_set,
                                  nactive, seed)
            # smoke は経路の確認なので、元の実行がまだ無い動作点でも通す
            allocations = (
                source_allocations(ROOT / 'result_logs', args.source_set,
                                   nactive, seed)[:2]
                if all((path / 'summary.json').exists() for path in sources)
                else [f'uniform{nactive // 2}', f'uniform{nactive}'])
        else:
            allocations = source_allocations(ROOT / 'result_logs',
                                             args.source_set, nactive, seed)
        reference = dense_reference(root)
        plan = {
            'layers': 2 if args.smoke else None,
            # slimpajama は7成分に最低1本ずつ配る。smoke でもそれ未満には落とせない
            'nsamples': 7 if args.smoke else NSAMPLES,
            # MMLU は57科目が別タスクなので、limit は**科目あたり**に効く
            'bench_limit': 2 if args.smoke else None,
            'bench_batch': 8 if args.smoke else BENCH_BATCH,
            'allocations': allocations,
            'reference': reference if reference.exists() else None,
        }
        log(f'  配分 {len(allocations)} 種: '
            + ', '.join(spec if spec.startswith('uniform') else '配分'
                        for spec in allocations))
        done = {'nactive': nactive, 'seed': seed,
                **bench_stage(root, args.source_set, nactive, seed, plan)}
        record['runs'].append(done)
        # 1本ずつ回しても --all で回しても、同じ名前の記録が残るようにする
        stem = ('' if args.source_set == 'main' else f'_{args.source_set}')
        summarize(root / f'exp17_stages{stem}_a{nactive}_seed{seed}.json',
                  {**record, 'runs': [done]})

    record['seconds'] = time.time() - started
    log(f'所要 {record["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
