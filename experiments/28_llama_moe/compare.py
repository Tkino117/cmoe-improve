"""28 の対応のある差を出す。GPU は要らない。

段1（分割だけの差）は**同じ配分どうし**、段2（手法の構成どうし）は提案手法を
対照に取る。層も再抽出も `cmoe run` の中の比較と同じものを通る。

配分の指定に使う ``@last`` は「その実行に渡された最後の --alloc」で、対照側では
beam 幅4、moe28 の v1 側でも beam 幅4 になる（run.py が uniform0 の次に足す）。

  uv run python experiments/28_llama_moe/compare.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run import DENSE_REFERENCE, POINTS, SEEDS, V2_SHARED   # noqa: E402

CONTROL = {6: 'result_logs/bench_slimpajama_seed{seed}',
           4: 'result_logs/bench_slimpajama_a4_seed{seed}'}


def dirs(template):
    return [template.format(seed=seed) for seed in SEEDS]


def run(title, base, base_alloc, cand, cand_alloc):
    missing = [row for row in base + cand
               if not os.path.exists(os.path.join(ROOT, row, 'summary.json'))]
    if missing:
        print(f'\n== {title} == （{missing[0]} が無いので飛ばす）')
        return
    command = ['uv', 'run', 'python',
               os.path.join(os.path.dirname(HERE), 'compare_runs.py'),
               '--base-alloc', base_alloc, '--cand-alloc', cand_alloc,
               '--reference', DENSE_REFERENCE]
    for row in base:
        command += ['--base', row]
    for row in cand:
        command += ['--cand', row]
    print(f'\n== {title} ==', flush=True)
    subprocess.run(command, cwd=ROOT, check=False)


def main():
    for label, nactive, _ in POINTS:
        control = dirs(CONTROL[nactive])
        v1 = dirs(f'result_logs/moe28_v1random_a{nactive}_seed{{seed}}')
        v2 = dirs(f'result_logs/moe28_v2_a{nactive}_seed{{seed}}')
        shared = f'uniform{V2_SHARED}'
        print(f'\n\n######## スパース率 {label}（A={nactive}） ########')

        print('\n---- 段1: 分割だけの差（同じ配分どうし。対照 = 現行 CMoE） ----')
        run(f'{label} uniform0: 現行 CMoE → LLaMA-MoE(Random)',
            control, 'uniform0', v1, 'uniform0')
        run(f'{label} {shared}: 現行 CMoE → LLaMA-MoE-v2',
            control, shared, v2, shared)
        run(f'{label} beam 幅4: 現行 CMoE → LLaMA-MoE(Random)',
            control, '@last', v1, '@last')

        print('\n---- 段2: 手法の構成どうし（対照 = 提案 beam 幅4） ----')
        run(f'{label} 提案 → LLaMA-MoE(Random) @ uniform0',
            control, '@last', v1, 'uniform0')
        run(f'{label} 提案 → LLaMA-MoE-v2 @ {shared}',
            control, '@last', v2, shared)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
