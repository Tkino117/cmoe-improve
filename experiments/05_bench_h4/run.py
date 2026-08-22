"""05 H4 — report/03 と同じ格子を、選択問題ベンチマークで測る。

report/03 は PPL だけで「探索した配分はどの一様配分よりも良い」を見た。ここは
**同じ配分・同じ seed・同じ校正**をベンチマークにかける。docs/03 の H4。

**探索は走らせない。** report/03 の18本の配分は
``result_logs/h1h3_pilot{,_seed1,_seed2}/h{2,3}_search_w{2,3,4}/search.json`` に
残っており、そこからベクトルを読み出して ``--alloc`` に渡す。探索がこの実験の
所要時間の大半だったので、これで12時間ぶんが要らなくなる。

dense の基準は **1回だけ測って全段で使い回す**。dense は seed にも配分にも校正にも
依らないので、段ごとに測り直すのは 8.8分 × 5 = 44分の丸損である。最初の段が測り、
残りは ``--bench-reference`` で取り込む。

段（1校正 × 1seed）ごとに別ディレクトリへ書き、終わった段は飛ばす。途中で落ちても
済んだ段は残り、同じコマンドで続きから走る。

  uv run python experiments/05_bench_h4/run.py --smoke   # 2層・問題を絞った確認
  uv run python experiments/05_bench_h4/run.py --seed 0  # seed 0（約3.9時間）
  uv run python experiments/05_bench_h4/run.py --seed 1
  uv run python experiments/05_bench_h4/run.py --seed 2
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

ROUTER = 'cmoe'
CALIBS = ('wikitext2', 'c4')
WIDTHS = (2, 3, 4)
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
# 対照。先頭が cmoe run の対応のある比較の基準になるので uniform3 を先に置く
# （report/03 と同じ並びにしておくと、PPL の表とベンチの表が同じ形になる）
BASELINE = 'uniform3'
UNIFORMS = [BASELINE] + [f'uniform{x}' for x in (0, 1, 2, 4, 5, 6)]

# report/03 の探索の置き場所。校正 wikitext2 は h2、c4 は h3 の段にある
SEARCH_ROOT = {0: 'h1h3_pilot', 1: 'h1h3_pilot_seed1', 2: 'h1h3_pilot_seed2'}
SEARCH_STAGE = {'wikitext2': 'h2', 'c4': 'h3'}

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


def finished(out_dir, expected):
    """終わった段なら記録を返す。無い・書きかけ・失敗込みなら None。

    完了印は「``summary`` があり、構成数が合い、``failures`` が無い」。ファイルの
    存在は開始直後から真になるので印にならない。ベンチも要求している段では
    ``bench_summary`` まで揃っていて初めて終わりとする。
    """
    payload = out_dir / 'summary.json'
    if not payload.exists():
        return None
    record = load(payload)
    if record.get('failures') or 'summary' not in record:
        return None
    if 'bench_summary' not in record:
        return None
    return record if len(record.get('runs', [])) == expected else None


def searched_vector(calib, width, seed):
    """report/03 が出した配分ベクトルを、``--alloc`` に渡す綴りで返す。"""
    path = (ROOT / 'result_logs' / SEARCH_ROOT[seed]
            / f'{SEARCH_STAGE[calib]}_search_w{width}' / 'search.json')
    if not path.exists():
        raise SystemExit(f'{path} が無い。report/03 の出力が要る')
    record = load(path)
    if 'allocation' not in record:
        raise SystemExit(f'{path} は終わっていない探索である')
    args = record.get('arguments', {})
    if (args.get('calib'), args.get('width'), args.get('seed')) != (calib, width, seed):
        raise SystemExit(
            f'{path} は calib={args.get("calib")} width={args.get("width")} '
            f'seed={args.get("seed")} の探索で、求めている '
            f'{calib}/{width}/{seed} と違う')
    return ','.join(str(value) for value in record['allocation']['values'])


def allocations_for(calib, seed, plan):
    """1段で測る配分。一様7種 + 探索3種（重複はまとめる）。

    ``--smoke`` は先頭 N 層しか変換しないので、探索配分もそこで切る。一様配分は
    層数から作られるので切る必要が無く、探索配分だけが 32 層ぶんの長さを持つ
    — ここで切らないと smoke が「配分は 32 層分、モデルは 2 層」で落ちる。
    """
    specs = list(plan['uniforms'])
    labels = {}
    for width in WIDTHS:
        spec = searched_vector(calib, width, seed)
        if plan['layers']:
            spec = ','.join(spec.split(',')[:plan['layers']])
        # 幅が違っても同じ配分に着くことがある。行を消さずに束ねる
        labels[spec] = (f'{labels[spec]}/{width}' if spec in labels
                        else f'beam 幅{width}')
        specs.append(spec)
    return list(dict.fromkeys(specs)), labels


def stage(calib, seed, out, reference, plan):
    """1校正 × 1seed。済んでいれば飛ばす。"""
    out_dir = out / f'{calib}_seed{seed}'
    specs, labels = allocations_for(calib, seed, plan)

    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', calib,
             '--datasets', DATASETS, '--bench',
             '--bench-batch-size', str(plan['bench_batch'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    # 基準がもう手元にあるなら測り直さない
    if reference is not None:
        argv += ['--bench-reference', str(reference)]
    argv += ['--out', str(out_dir)]

    record = finished(out_dir, len(specs))
    if record is None:
        if out_dir.exists():
            payload = out_dir / 'summary.json'
            if payload.exists():
                differences = same_experiment(load(payload), argv)
                if differences:
                    raise SystemExit(
                        f'{out_dir} は違う実験の出力である'
                        f'（{", ".join(differences)}）。別の --out を指定するか、'
                        'そのディレクトリを退けること')
            retire(out_dir)
        run_cli(argv)
        record = finished(out_dir, len(specs))
        if record is None:
            raise SystemExit(f'{out_dir} に終わった段が残らなかった')
    else:
        log(f'  {out_dir.name}: 済み')
    differences = same_experiment(record, argv)
    if differences:
        raise SystemExit(
            f'{out_dir} は違う実験の出力である（{", ".join(differences)}）')
    return {'calib': calib, 'seed': seed, 'dir': str(out_dir.relative_to(ROOT)),
            'labels': labels, 'specs': specs}


def find_reference(out):
    """すでに測ってある dense の基準を探す。

    seed ごとに起動を分ける（``--seed 1`` を後日打つ）使い方でも測り直しが
    起きないように、出力先の下にある**終わった段**の基準を拾う。書きかけの段の
    ものは拾わない — 途中で落ちた実行のファイルを基準にすると、そこから先の
    ``ref_kl`` がすべて別物になる。
    """
    for path in sorted(out.glob('*/bench/dense.json')):
        payload = path.parent.parent / 'summary.json'
        if not payload.exists():
            continue
        record = load(payload)
        if 'bench_summary' in record and not record.get('failures'):
            return path.resolve()
    return None


def summarize(out, stages, seconds):
    """段の一覧と、測ったコードのコミットを残す。

    数値そのものは集計しない。段ごとの ``summary.json`` と ``bench/*.json`` に
    全部あり、seed をまたいだまとめは ``summarize_seeds.py`` が
    GPU なしで作る（指標をあとから足せるのが、この分け方の目的である）。
    """
    payload = {
        'stages': stages,
        'seconds': seconds,
        'commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        'dirty': bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
            text=True).stdout.strip()),
    }
    path = out / 'stages.json'
    with path.open('w') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calibs', default=','.join(CALIBS),
                        help='校正セット（既定は wikitext2,c4）')
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・タスクあたり8問。経路の確認')
    parser.add_argument('--uniforms', default=','.join(UNIFORMS),
                        help='一様配分の対照（先頭が対応のある比較の基準）')
    args = parser.parse_args(argv)

    uniforms = [name.strip() for name in args.uniforms.split(',') if name.strip()]
    if args.smoke and args.uniforms == ','.join(UNIFORMS):
        # smoke で見たいのは経路であって表ではない。PPL は評価セット全体を走る
        # ので構成数がそのまま時間になる。一様は対照1つと比較1つで足りる
        uniforms = [BASELINE, 'uniform4']
    plan = {
        'layers': 2 if args.smoke else None,
        'bench_limit': 8 if args.smoke else None,
        'bench_batch': 8 if args.smoke else BENCH_BATCH,
        'uniforms': uniforms,
    }
    suffix = '_smoke' if args.smoke else ''
    out = Path(args.out) if args.out else ROOT / f'result_logs/bench_h4{suffix}'
    out.mkdir(parents=True, exist_ok=True)

    calibs = [name.strip() for name in args.calibs.split(',') if name.strip()]
    log(f'seed {args.seed} / 校正 {",".join(calibs)} / '
        f'配分 {len(plan["uniforms"]) + len(WIDTHS)} 種'
        + ('（smoke）' if args.smoke else ''))
    log(f'出力 {out}')

    started = time.time()
    stages = []
    reference = find_reference(out)
    if reference is not None:
        log(f'dense の基準は測り済み: {reference}')
    for calib in calibs:
        log(f'\n=== {calib} / seed {args.seed} ===')
        record = stage(calib, args.seed, out, reference, plan)
        stages.append(record)
        if reference is None:
            # 最初の段が測った dense を、以降の段が使い回す
            reference = (ROOT / record['dir'] / 'bench' / 'dense.json').resolve()
            log(f'  以降の段は {reference} を基準に使う')

    summarize(out, stages, time.time() - started)
    log(f'所要 {(time.time() - started) / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
