"""28 の対応のある差を出す。GPU は要らない。

段1（分割だけの差）は**同じ配分どうし**、段2（手法の構成どうし）は提案手法を
対照に取る。層も再抽出も `cmoe run` の中の比較と同じものを通る。

配分の指定に使う ``@last`` は「その実行に渡された最後の --alloc」で、対照側では
beam 幅4、moe28 の v1 側でも beam 幅4 になる（run.py が uniform0 の次に足す）。

  uv run python experiments/28_llama_moe/compare.py
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run import (DENSE_REFERENCES, MODELS, POINTS, SEEDS,   # noqa: E402
                 V2_SHARED, model_tag)
from table import control_template                          # noqa: E402


def dirs(template, seeds):
    return [template.format(seed=seed) for seed in seeds]


def run(title, base, base_alloc, cand, cand_alloc, reference):
    missing = [row for row in base + cand
               if not os.path.exists(os.path.join(ROOT, row, 'summary.json'))]
    if missing:
        print(f'\n== {title} == （{missing[0]} が無いので飛ばす）')
        return
    command = ['uv', 'run', 'python',
               os.path.join(os.path.dirname(HERE), 'compare_runs.py'),
               '--base-alloc', base_alloc, '--cand-alloc', cand_alloc,
               '--reference', reference]
    for row in base:
        command += ['--base', row]
    for row in cand:
        command += ['--cand', row]
    print(f'\n== {title} ==', flush=True)
    subprocess.run(command, cwd=ROOT, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='llama2-7b', choices=sorted(MODELS))
    parser.add_argument('--seeds', default=None, help='既定は 0,1,2')
    args = parser.parse_args()
    tag = model_tag(args.model)
    seeds = (tuple(int(x) for x in args.seeds.split(',')) if args.seeds
             else SEEDS)
    reference = DENSE_REFERENCES[args.model]

    for label, nactive, _ in POINTS:
        control = dirs(control_template(tag, nactive, seeds), seeds)
        v1 = dirs(f'result_logs/moe28_v1random{tag}_a{nactive}_seed{{seed}}',
                  seeds)
        v2 = dirs(f'result_logs/moe28_v2{tag}_a{nactive}_seed{{seed}}', seeds)
        # EW-rule（experiments/21）。run の最後の --alloc が規則の出した配分で、
        # 先頭は同じ run の中の一様の対照である
        suffix = '' if nactive == 6 else f'_a{nactive}'
        ew = dirs(f'result_logs/ew_bench{tag}{suffix}_seed{{seed}}', seeds)
        shared = f'uniform{V2_SHARED}'
        print(f'\n\n######## スパース率 {label}（A={nactive}） ########')
        print(f'（対照 = {control[0]} …）')

        print('\n---- 段1: 分割だけの差（同じ配分どうし。対照 = 現行 CMoE） ----')
        run(f'{label} uniform0: 現行 CMoE → LLaMA-MoE(Random)',
            control, 'uniform0', v1, 'uniform0', reference)
        run(f'{label} {shared}: 現行 CMoE → LLaMA-MoE-v2',
            control, shared, v2, shared, reference)
        run(f'{label} beam 幅4: 現行 CMoE → LLaMA-MoE(Random)',
            control, '@last', v1, '@last', reference)

        print('\n---- 段2: 手法の構成どうし（対照 = 提案 beam 幅4） ----')
        run(f'{label} 提案 → LLaMA-MoE(Random) @ uniform0',
            control, '@last', v1, 'uniform0', reference)
        run(f'{label} 提案 → LLaMA-MoE-v2 @ {shared}',
            control, '@last', v2, shared, reference)
        run(f'{label} 提案 → EW-rule',
            control, '@last', ew, '@last', reference)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
