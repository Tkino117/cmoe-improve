"""30 層を独立に決める配分と、前から順に決める幅1 を PPL で比べる。

論文の査読想定の指摘 —「層間の依存を考慮したこと」の効果が切り分けられていない —
への対照である。採点オラクルは両方とも `suffix_kl`（出力分布の KL）で、評価回数も
同じ（層数 × 候補数）。違いは次の層へ渡す状態だけ:

  幅1（greedy）   決めた x で変換した状態を渡す。層 ℓ は先行層の変換の上で測る
  independent     dense のまま渡す。層 ℓ は「層 ℓ だけを変換したモデル」で測る

段は3つで、**済んだ段は飛ばす**。

  1. 探索 independent      → result_logs/slimpajama<_a4>_indep_n16_seed<N>
  2. 探索 幅1（無ければ）  → result_logs/slimpajama<_a4>_w1_n16_seed<N>
  3. PPL                   → result_logs/ppl30<_a4>_seed<N>
     uniform{A/2}（CMoE）・幅1・幅2（report/27 の配分）・independent を同じ実行で測る

幅2 の配分は report/27 の別マシンで探したもの（`result_logs/exp27_imported/`）で、
**測定はここで引き直す**。探索は機械が違うと同じ配分にならない（同じ seed でも
ローカルの幅2 と食い違う）ので、主比較は同じ機械で探した independent 対 幅1 である。

  uv run python experiments/30_independent_kl/run.py --smoke
  uv run python experiments/30_independent_kl/run.py --seed 0 --nactive 6
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

CALIB = 'slimpajama'
NSAMPLES = 16
ORACLE = 'suffix_kl'
ROUTER = 'cmoe'
NEXPERTS = 8
DATASETS = 'wikitext2,c4-new'
IMPORTED = LOGS / 'exp27_imported'

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


def suffix(nactive):
    """出力名の動作点。A=6 は無印、それ以外は _a<A>（report/05〜27 と同じ）。"""
    return '' if nactive == 6 else f'_a{nactive}'


def search_stage(seed, plan, root, label, search_args):
    out_dir = root / f'{CALIB}{suffix(plan["nactive"])}_{label}_n{plan["nsamples"]}_seed{seed}'
    argv = ['search', '--calib', CALIB, '--nsamples', str(plan['nsamples']),
            '--oracle', ORACLE, '--seed', str(seed),
            '--nexperts', str(NEXPERTS), '--nactive', str(plan['nactive']),
            *search_args]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return 'allocation' in record and 'recheck' in record

    record = ensure(out_dir, 'search.json', argv, is_finished)
    log(f'  {label}: score {record["score"]:.6e} '
        f'平均 x={record["allocation"]["mean_x"]:.3f} '
        f'({record["calls"]} 回 / {record["seconds"] / 60:.1f}分)')
    return record, out_dir


def imported_w2(seed, plan, calibration_hash):
    """report/27 の幅2 の配分。校正トークンが同じことを確かめてから使う。"""
    path = IMPORTED / f'{CALIB}{suffix(plan["nactive"])}_w2_n{plan["nsamples"]}_seed{seed}' / 'search.json'
    if not path.exists():
        raise SystemExit(f'{path} が無い。report/27 の幅2 の配分が要る')
    record = load(path)
    arguments = record['arguments']
    expected = {'model': 'meta-llama/Llama-2-7b-hf', 'calib': CALIB,
                'nsamples': plan['nsamples'], 'seed': seed, 'oracle': ORACLE,
                'nexperts': NEXPERTS, 'nactive': plan['nactive'],
                'search': 'beam', 'width': 2}
    wrong = [key for key, value in expected.items() if arguments.get(key) != value]
    if wrong:
        raise SystemExit(f'{path} は幅2 の同じ設定の探索ではない（{", ".join(wrong)}）')
    if record['calibration']['token_hash'] != calibration_hash:
        raise SystemExit(f'{path} の校正トークンが、ここで探した independent と違う')
    return record


def ppl_stage(seed, plan, root, specs):
    out_dir = root / f'ppl30{suffix(plan["nactive"])}_seed{seed}'
    specs = list(dict.fromkeys(specs))
    argv = ['run']
    for spec in specs:
        argv += ['--alloc', spec]
    argv += ['--router', ROUTER, '--seeds', str(seed), '--calib', CALIB,
             '--nsamples', str(plan['nsamples']),
             '--nexperts', str(NEXPERTS), '--nactive', str(plan['nactive']),
             '--datasets', DATASETS]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += ['--out', str(out_dir)]

    def is_finished(record):
        return (not record.get('failures') and 'summary' in record
                and len(record.get('runs', [])) == len(specs))

    ensure(out_dir, 'summary.json', argv, is_finished)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nactive', type=int, default=6,
                        help='1トークンあたりに走る expert 数 A（6 で 25%%、4 で 50%%）')
    parser.add_argument('--smoke', action='store_true', help='2層・7本。配線の確認')
    args = parser.parse_args(argv)

    plan = {'nactive': args.nactive,
            'layers': 2 if args.smoke else None,
            # slimpajama は7成分に最低1本ずつ配る
            'nsamples': 7 if args.smoke else NSAMPLES}
    root = LOGS / 'exp30_smoke' if args.smoke else LOGS
    root.mkdir(parents=True, exist_ok=True)
    sparsity = 100 * (NEXPERTS - args.nactive) // NEXPERTS
    log(f'seed {args.seed} / N={NEXPERTS} A={args.nactive}（スパース率 {sparsity}%）'
        + ('（smoke）' if args.smoke else ''))

    started = time.time()
    stages = {'seed': args.seed, 'nactive': args.nactive, 'smoke': args.smoke}

    log('\n=== 段1 探索 independent ===')
    indep, indep_dir = search_stage(args.seed, plan, root, 'indep',
                                    ['--search', 'independent'])
    log('\n=== 段2 探索 幅1 ===')
    greedy, greedy_dir = search_stage(args.seed, plan, root, 'w1',
                                      ['--search', 'beam', '--width', '1'])
    if greedy['calibration']['token_hash'] != indep['calibration']['token_hash']:
        raise SystemExit('幅1 と independent の校正トークンが違う。比べられない')

    uniform = f'uniform{args.nactive // 2}'
    specs = [uniform]
    vectors = {'indep': indep['allocation']['values'],
               'w1': greedy['allocation']['values']}
    if not args.smoke:
        w2 = imported_w2(args.seed, plan, indep['calibration']['token_hash'])
        vectors['w2'] = w2['allocation']['values']
    specs += [','.join(str(x) for x in values) for values in vectors.values()]

    if args.smoke:
        # 2層の接頭辞は run --alloc に渡せない（全層ぶんを要求する）。配線はここまで
        log('\n（smoke は探索まで。PPL の段は全層の配分が要る）')
    else:
        log('\n=== 段3 PPL ===')
        stages['ppl_dir'] = str(ppl_stage(args.seed, plan, root, specs).relative_to(ROOT))

    stages.update({
        'uniform': uniform, 'vectors': vectors,
        'search_dirs': {'indep': str(indep_dir.relative_to(ROOT)),
                        'w1': str(greedy_dir.relative_to(ROOT))},
        'scores': {'indep': indep['score'], 'w1': greedy['score']},
        'seconds': time.time() - started,
        'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    })
    path = root / f'exp30_stages{suffix(args.nactive)}_seed{args.seed}.json'
    with path.open('w') as handle:
        json.dump(stages, handle, indent=1, ensure_ascii=False)
    log(f'\n段の一覧を {path} に書いた / 所要 {stages["seconds"] / 60:.1f}分')
    return 0


if __name__ == '__main__':
    sys.exit(main())
