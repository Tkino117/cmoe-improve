"""28 LLaMA-MoE / LLaMA-MoE-v2 の分割規則を対照として足す。

CMoE 最新版 (ACL 2026) Table 1 が並べている MoE Restructuring の比較手法である。
**どちらも提案手法ではない。**

**移すのは分割（軸3）だけ。** 両手法は学習前提（v1 は継続事前学習 200B トークン、
v2 は post-training 約7B トークン）で、分割直後の状態は原典の評価対象ではない。
CMoE 論文も分割だけを再実装して学習量を揃えており（§5.1・Table 6 の
"Split-only; training time not included"）、ここはそれを**追加学習なし**の土俵
（report/25 と同じ）に置いたものである。

  1. probe  ``probe.py``   v2 の層ごと・クラスタごとの重要度 → result_logs/llama_moe_v2_probe_seed<N>
  2. bench  ``cmoe run``   PPL と選択問題                    → result_logs/moe28_*

**対照は測り直さない。** 現行 CMoE の分割で同じ配分を測ったものが
``bench_slimpajama{,_a4}_seed<N>`` に既にある（``uniform0`` / ``uniform1`` /
beam 幅4）。分割だけを差し替えた比較はそこと突き合わせて作る。

``--method control`` は現行 CMoE の分割を、比べる3つの配分だけ同じ run で
測り直す段である。report/26 は既存の ``bench_slimpajama*`` を対照に流用したが、
あれは測ったマシンが違いうる — 同じ土俵で読みたいときはこの段を通す。

  uv run python experiments/28_llama_moe/run.py --smoke
  uv run python experiments/28_llama_moe/run.py
  uv run python experiments/28_llama_moe/run.py --model mistral-7b --method control,v1,v2
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
# 短い札 → HuggingFace の名前。experiments/25 と同じ辞書である
MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
}
# 出力先の名前に札が入らないモデル。report/26 の頃の名前を保つ
UNTAGGED_MODEL = 'llama2-7b'
NSAMPLES = 16
NEXPERTS = 8
DATASETS = 'wikitext2,c4-new'
BENCH_BATCH = 32
SEEDS = (0, 1, 2)
# (スパース率のラベル, A, 探索ディレクトリの接尾辞)
POINTS = (('25%', 6, ''), ('50%', 4, '_a4'))
# v2 の residual 数。原典の主構成 1+7top1 に対応する。**クラスタ数 = 8 - これ**
V2_SHARED = 1
# 取り込む dense。**モデルごとに測ったものを使う。** Llama は report/04 の
# ``bench_h4``（report/26 がこれで測った）、Mistral は experiments/25 の
# ``--check`` が測ったもの。GPU をまたぐと効果量と同じ桁の差が出かねないので、
# 基準だけ別のハードウェアのものにしない
DENSE_REFERENCES = {
    'llama2-7b': 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json',
    'mistral-7b': ('result_logs/exp25_check/bench_slimpajama_mistral-7b_seed0'
                   '/bench/dense.json'),
}
NOT_IDENTITY = {'out', 'batch_chunk', 'token_chunk', 'search_token_chunk',
                'bench_reference', 'bench_cache_dir', 'carve_scores'}


def log(message=''):
    print(message, flush=True)


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


def run_command(command):
    log(f'$ {" ".join(str(part) for part in command)}')
    subprocess.run(command, cwd=ROOT, check=True)


def model_tag(model_key):
    """出力先に入るモデルの札。Llama-2-7b だけ空である。"""
    return '' if model_key == UNTAGGED_MODEL else f'_{model_key}'


def beam4_vector(tag, suffix, seed):
    """提案手法（beam 幅4）が出した配分。探索の出力から引く。

    bench のディレクトリには幅2/3/4 が並んでいるが、幅3 と幅4 が同じベクトルに
    なる seed があり（A=4 seed 0）、位置では指せない。
    """
    path = (ROOT /
            f'result_logs/slimpajama{tag}{suffix}_w4_n{NSAMPLES}_seed{seed}'
            '/search.json')
    if not path.exists():
        raise SystemExit(f'{path} が無い')
    return ','.join(str(value) for value in load(path)['allocation']['values'])


# -- 段1: v2 の重要度プローブ ---------------------------------------------

def probe_stage(seed, plan):
    """クラスタ数は ``NEXPERTS - V2_SHARED`` で A に依らないので、seed ごとに1本。"""
    out = (ROOT / f'result_logs/llama_moe_v2_probe{plan["tag"]}'
           f'{plan["suffix"]}_seed{seed}')
    wanted = {'model': plan['model'], 'calib': CALIB, 'seed': seed,
              'nsamples': plan['nsamples'], 'seqlen': 2048,
              'nexperts': NEXPERTS, 'nshared': V2_SHARED,
              'layers': plan['layers'], 'gate_tokens': 4096}
    command = ['uv', 'run', 'python', str(HERE / 'probe.py'),
               '--model', plan['model'],
               '--seed', str(seed), '--calib', CALIB,
               '--nsamples', str(plan['nsamples']),
               '--nexperts', str(NEXPERTS), '--nshared', str(V2_SHARED),
               '--out', str(out.relative_to(ROOT))]
    if plan['layers']:
        command += ['--layers', str(plan['layers'])]

    payload = out / 'scores.json'
    record = load(payload) if payload.exists() else None
    finished = record is not None and len(record.get('scores', [])) > 0
    if finished:
        log(f'  {out.name}: 済み')
    else:
        if out.exists():
            retire(out)
        run_command(command)
        record = load(payload)
    given = record.get('arguments', {})
    gaps = [key for key in sorted(wanted) if given.get(key) != wanted[key]]
    if gaps:
        raise SystemExit(f'{out} は違う設定のプローブである（{", ".join(gaps)}）')
    return out


# -- 段2: 変換して測る ----------------------------------------------------

def cli_differences(record, argv):
    given = record.get('arguments', {})
    wanted = vars(build_parser().parse_args(argv))
    return [key for key in sorted(wanted)
            if key not in NOT_IDENTITY and given.get(key) != wanted[key]]


def bench_stage(name, carver, allocs, nactive, seed, plan, scores=None):
    out = (ROOT / f'result_logs/moe28_{name}{plan["tag"]}_a{nactive}'
           f'{plan["suffix"]}_seed{seed}')
    argv = [
        'run',
        '--model', plan['model'],
        '--carver', carver,
        '--calib', CALIB,
        '--nsamples', str(plan['nsamples']),
        '--nexperts', str(NEXPERTS),
        '--nactive', str(nactive),
        '--seeds', str(seed),
        '--datasets', DATASETS,
        '--bench',
        '--bench-batch-size', str(BENCH_BATCH),
        '--out', str(out.relative_to(ROOT)),
    ]
    for alloc in allocs:
        argv += ['--alloc', alloc]
    if scores is not None:
        argv += ['--carve-scores', str(Path(scores).relative_to(ROOT) / 'scores.json')]
    if plan['layers']:
        argv += ['--layers', str(plan['layers'])]
    argv += (['--no-bench-dense'] if plan['bench_limit']
             else ['--bench-reference', plan['dense']])
    if plan['bench_limit']:
        argv += ['--bench-limit', str(plan['bench_limit'])]

    payload = out / 'summary.json'
    record = load(payload) if payload.exists() else None
    finished = record is not None and len(record.get('runs', [])) == len(allocs)
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
        if record is None or len(record.get('runs', [])) != len(allocs):
            raise SystemExit(f'{out} に {len(allocs)} 構成が残らなかった')
    gaps = cli_differences(record, argv)
    if gaps:
        raise SystemExit(f'{out} は違う実験の出力である（{", ".join(gaps)}）')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='2層・8問で経路だけ通す')
    parser.add_argument('--method', default='v1,v2',
                        help='control,v1,v2 から選ぶ。control は現行 CMoE の'
                             '分割を同じマシンで測り直す段である')
    parser.add_argument('--model', default='llama2-7b', choices=sorted(MODELS))
    parser.add_argument('--seeds', default=None,
                        help='既定は 0,1,2')
    parser.add_argument('--nactives', default=None,
                        help='回す動作点。既定は 6,4')
    args = parser.parse_args()

    tag = model_tag(args.model)
    seeds = (tuple(int(x) for x in args.seeds.split(','))
             if args.seeds else SEEDS)
    plan = {'model': MODELS[args.model], 'seeds': seeds, 'nsamples': NSAMPLES,
            'layers': None, 'bench_limit': None, 'suffix': '', 'tag': tag,
            'dense': DENSE_REFERENCES[args.model]}
    if args.smoke:
        plan.update(seeds=(0,), layers=2, bench_limit=8, suffix='_smoke')
    methods = args.method.split(',')
    log(f'{args.model}（{plan["model"]}）/ seed '
        f'{",".join(str(s) for s in plan["seeds"])} / 手法 {args.method}')

    points = POINTS
    if args.nactives:
        wanted = {int(x) for x in args.nactives.split(',')}
        points = tuple(row for row in POINTS if row[1] in wanted)
        if not points:
            raise SystemExit(f'{args.nactives} に当たる動作点が無い')
    for label, nactive, search_suffix in points:
        for seed in plan['seeds']:
            log()
            log(f'== スパース率 {label} (A={nactive}) / seed {seed} ==')
            if 'control' in methods:
                # 現行 CMoE の分割を、**同じマシン・同じ run で**測り直す段。
                # report/26 は既存の ``bench_slimpajama*`` を対照に流用したが、
                # あれはモデルによっては別マシンの測定である。比べる3つの配分
                # （v1 の uniform0 / v2 の uniform1 / 提案 幅4）だけを測る
                allocs = ['uniform0', f'uniform{V2_SHARED}']
                if not plan['layers']:
                    allocs.append(beam4_vector(tag, search_suffix, seed))
                bench_stage('control', 'cmoe', allocs, nactive, seed, plan)
            if 'v1' in methods:
                # 原典の構造は shared 無し（uniform0）。提案と同じ配分の行も足して、
                # 「分割だけの差」と「配分も違う差」を分ける
                allocs = ['uniform0']
                if not plan['layers']:
                    allocs.append(beam4_vector(tag, search_suffix, seed))
                bench_stage('v1random', 'llama_moe_random', allocs, nactive,
                            seed, plan)
            if 'v2' in methods:
                probe = probe_stage(seed, plan)
                # residual 1本。**クラスタ数 = routed 数**なので x は動かせない
                bench_stage('v2', 'llama_moe_v2', [f'uniform{V2_SHARED}'],
                            nactive, seed, plan, scores=probe)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
