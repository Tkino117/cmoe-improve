"""27 静的な構造化プルーニングの対照を足す（FLAP / LLM-Pruner）。

ExpertWeaver Table 2 が並べている training-free の比較手法を、この基盤の評価に
載せる。**どちらも提案手法ではない。**

[report/21](../../report/21_ew-rule-baseline.md)・[report/22](../../report/22_alloc-baselines.md)
の対照は、どれも「MoE 変換のうち配分軸だけを差し替えたもの」だった。ここは
初めて MoE 変換を通らない対照で、FFN ニューロンと attention ヘッドを**恒久的に
削除**して dense のまま測る。したがって揃うのは**トークンあたりの活性パラメータ
（＝ FLOPs）だけ**で、メモリは揃わない。

**スパース率を2つの土俵で測る。** この基盤の「25%」は FFN ニューロンの比
（N=8 / A=6）で、ブロック全体では 16.71% にしかならない。一方 FLAP の
``pruning_ratio`` は attn+FFN 全体に対する比である。同じ「25%」と書くと対照だけが
1.5倍削ることになるので、両方を測る。

  scope=block  原典の定義そのまま。ExpertWeaver Table 2 と同じ土俵
  scope=mlp    FFN だけを刈り、活性パラメータを提案手法に厳密に揃えた土俵

**校正は slimpajama、本数と長さは原典の既定のまま**（FLAP 2048本×128、
LLM-Pruner 10本×64）。ソースだけを提案手法と揃えてある。対照から校正を取り上げ
ないための選択で、提案手法が最も少ない校正（16本×2048）で戦っていることは
レポート側に書く。

  1. prune  ``cmoe prune``  4本（手法2 × scope2）× スパース率2 × seed3 = 24構成
  2. 比較   ``compare_runs.py``  提案（beam 幅4）と EW-rule に対する対応のある差

済んだ段は飛ばす。GPU が落ちたらそのまま同じコマンドを叩き直せばよい。

  uv run python experiments/27_static_pruning/run.py --smoke
  uv run python experiments/27_static_pruning/run.py
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from cmoe.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CALIB = 'slimpajama'
MODEL = 'meta-llama/Llama-2-7b-hf'
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
SEEDS = (0, 1, 2)
SPARSITIES = ('0.25', '0.5')
METHODS = ('flap', 'llm_pruner')
SCOPES = ('block', 'mlp')
# report/04 が測った dense。experiments/06・07・21・22 と同じものを取り込む
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
# 結果を変えない引数。同じ実験かの判定から外す（experiments/22 と同じ集合）
NOT_IDENTITY = {'out', 'batch_chunk', 'bench_reference', 'bench_cache_dir'}

# 対応のある差を取る相手。report/21 と同じ出どころで、動作点ごとに1組
# 探索が出した配分は seed ごとに違うベクトルなので、``@last``（その実行に渡された
# 最後の --alloc = 幅4）で指す。experiments/06・07 が幅2/3/4 をこの順で渡している
CONTROLS = {
    '0.25': {'dirs': [f'result_logs/bench_slimpajama_seed{s}' for s in SEEDS],
             'alloc': '@last', 'label': '提案（beam 幅4）'},
    '0.5': {'dirs': [f'result_logs/bench_slimpajama_a4_seed{s}' for s in SEEDS],
            'alloc': '@last', 'label': '提案（beam 幅4）'},
}


def log(message=''):
    print(message, flush=True)


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def retire(path):
    """書きかけを退避する。消さない（落ちた跡は原因調べに要る）。"""
    for index in itertools.count(1):
        target = path.with_name(f'{path.name}.partial{index}')
        if not target.exists():
            path.rename(target)
            log(f'  書きかけの {path.name} を {target.name} へ退避した')
            return target


def run_command(command):
    log(f'$ {" ".join(str(part) for part in command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def cli_differences(record, argv):
    wanted = vars(build_parser().parse_args(argv))
    given = record.get('arguments', {})
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def prune_stage(method, scope, plan):
    """1手法 × 1 scope。スパース率2種 × seed3本を1コマンドで通す。"""
    out = ROOT / f'result_logs/prune27_{method}_{scope}{plan["suffix"]}'
    argv = [
        'prune',
        '--model', plan['model'],
        '--method', method,
        '--scope', scope,
        '--sparsity', ','.join(plan['sparsities']),
        '--calib', CALIB,
        '--seeds', ','.join(str(seed) for seed in plan['seeds']),
        '--datasets', DATASETS,
        '--bench',
        '--bench-batch-size', str(BENCH_BATCH),
        '--out', str(out.relative_to(ROOT)),
    ]
    # smoke は --bench-limit を付けるので、全件で測った dense とは対応が取れない
    argv += (['--no-bench-dense'] if plan['bench_limit']
             else ['--bench-reference', DENSE_REFERENCE])
    if plan['calib_samples']:
        argv += ['--calib-samples', str(plan['calib_samples'])]
    if plan['calib_seqlen']:
        argv += ['--calib-seqlen', str(plan['calib_seqlen'])]
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]
    expected = len(plan['sparsities']) * len(plan['seeds'])

    payload = out / 'summary.json'
    record = load(payload) if payload.exists() else None
    finished = record is not None and len(record.get('runs', [])) == expected
    if finished:
        log(f'  {out.name}: 済み')
    else:
        if out.exists():
            if record is not None and cli_differences(record, argv):
                raise SystemExit(
                    f'{out} は違う実験の出力である'
                    f'（{", ".join(cli_differences(record, argv))}）')
            retire(out)
        run_command(['uv', 'run', 'cmoe'] + argv)
        record = load(payload) if payload.exists() else None
        if record is None or len(record.get('runs', [])) != expected:
            raise SystemExit(f'{out} に {expected} 構成が残らなかった')
    gaps = cli_differences(record, argv)
    if gaps:
        raise SystemExit(f'{out} は違う実験の出力である（{", ".join(gaps)}）')
    return out


def compare_stage(prune_dirs, plan):
    """提案手法との対応のある差。層も再抽出も CLI と同じものを通る。"""
    for sparsity in plan['sparsities']:
        control = CONTROLS.get(sparsity)
        if control is None:
            log(f'  スパース率 {sparsity} に対照が無いので比較は飛ばす')
            continue
        missing = [row for row in control['dirs'] if not (ROOT / row).exists()]
        if missing:
            log(f'  対照 {missing} が無いので比較は飛ばす')
            continue
        for (method, scope), out in prune_dirs.items():
            label = f'{method}@{scope}/{float(sparsity):g}'
            command = ['uv', 'run', 'python', str(HERE.parent / 'compare_runs.py'),
                       '--cand-alloc', label, '--base-alloc', control['alloc'],
                       '--reference', DENSE_REFERENCE]
            for row in control['dirs']:
                command += ['--base', row]
            command += ['--cand', str(out.relative_to(ROOT))]
            log()
            log(f'== {control["label"]} vs {label}（スパース率 {sparsity}）==')
            subprocess.run(command, cwd=ROOT, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='2構成・8問で経路だけ通す')
    parser.add_argument('--method', default=','.join(METHODS))
    parser.add_argument('--scope', default=','.join(SCOPES))
    parser.add_argument('--calib-match', action='store_true',
                        help='校正の量を提案手法（16本×2048）に揃えて測り直す')
    parser.add_argument('--no-compare', action='store_true')
    args = parser.parse_args()

    plan = {
        'model': MODEL, 'seeds': SEEDS, 'sparsities': SPARSITIES,
        'calib_samples': None, 'calib_seqlen': None, 'bench_limit': None,
        'suffix': '',
    }
    if args.smoke:
        plan.update(seeds=(0,), sparsities=('0.25',), calib_samples=16,
                    bench_limit=8, suffix='_smoke')
    if args.calib_match:
        # 校正の量だけを提案手法に揃える。主表の FLAP は原典の既定（2048本×128
        # = 262k トークン）で、提案手法は 16本×2048 = 32k トークンしか読まない。
        # 主表で FLAP が勝ったとき、その差が「手法」なのか「校正が8倍」なのかを
        # 分けるのがこの段である。
        #
        # 窓の長さが変わっても比較は崩れない。FLAP の fluc は系列ごとの総和を
        # 系列数で割る形なので、窓を長くすると全チャネルが同じ倍率で大きくなる。
        # 層ごとの標準化がその共通倍率を落とすので、AL-AM のしきい値は変わらない
        plan.update(calib_samples=16, calib_seqlen=2048, suffix='_calib16')

    prune_dirs = {}
    for method in args.method.split(','):
        for scope in args.scope.split(','):
            log()
            log(f'== prune {method} / scope={scope} ==')
            prune_dirs[(method, scope)] = prune_stage(method, scope, plan)

    if not args.no_compare and not args.smoke:
        compare_stage(prune_dirs, plan)
    if args.calib_match:
        log()
        log('校正を揃えた版である。主表は接尾辞の無いディレクトリを見ること')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
