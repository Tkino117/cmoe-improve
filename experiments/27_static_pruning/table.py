"""27 の主表を作る。**校正データを揃えた版だけ**を並べる。GPU は要らない。

行は 提案（beam 幅2/3/4）と静的プルーニング3本、列は PPL・acc・gold_nll。
スパース率ごとに別の表にする。

**実効ブロック削減率を必ず添える。** FLAP の「本来の定義」の行だけは削減量が
他の行より多い（attn+FFN 全体の比なので）。この列が無いと、行を横並びに読んで
しまう。

提案手法の幅ごとのベクトルは、その幅の探索の ``search.json`` から引く。bench の
ディレクトリには幅2/3/4 が並んでいるが、幅3 と幅4 が同じベクトルになる seed が
あり（A=4 seed 0）、位置では指せない。

  uv run python experiments/27_static_pruning/table.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import compare_runs                                          # noqa: E402

from cmoe.eval import bench, bench_stats                     # noqa: E402
from cmoe.prune.base import moe_activated_sparsity           # noqa: E402

# Llama-2-7B。提案手法の行の実効削減率を出すためだけに要る
SHAPE = {'hidden_size': 4096, 'intermediate_size': 11008}

SEEDS = (0, 1, 2)
WIDTHS = (2, 3, 4)
DENSE_REFERENCE = 'result_logs/bench_h4/wikitext2_seed0/bench/dense.json'
# 変換前の PPL。トークンハッシュが評価セットと一致することを確かめてから使う
DENSE_PPL = 'result_logs/dense_ppl_llama-2-7b-hf/dense_ppl.json'
PPL_DATASET = 'wikitext2'

# スパース率 -> (探索ディレクトリの接尾辞, bench ディレクトリの雛形)
POINTS = {
    '25%': ('', 'result_logs/bench_slimpajama_seed{seed}', 6),
    '50%': ('_a4', 'result_logs/bench_slimpajama_a4_seed{seed}', 4),
}
# 静的プルーニングの行。すべて校正を提案手法に揃えた版（16本×2048）
PRUNE_ROWS = (
    ('FLAP（FFN のみ）', 'result_logs/prune27_flap_mlp_calib16', 'flap@mlp/{value}'),
    ('FLAP（本来の定義）', 'result_logs/prune27_flap_block_calib16',
     'flap@block/{value}'),
    ('LLM-Pruner', 'result_logs/prune27_llm_pruner_mlp_calib16',
     'llm_pruner@mlp/{value}'),
)
VALUES = {'25%': '0.25', '50%': '0.5'}


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def search_vector(suffix, width, seed):
    record = load(f'result_logs/slimpajama{suffix}_w{width}_n16_seed{seed}/search.json')
    return ','.join(str(value) for value in record['allocation']['values'])


def measure(runs, reference):
    """(PPL, acc, gold_nll, 実効削減率)。すべて seed 平均。"""
    ppl = [compare_runs.run_ppl(run, 'cmoe', PPL_DATASET)['ppl']
           for _, run in runs.values()]
    per_seed = [bench_stats.summarize(bench.load_samples(os.path.join(
        ROOT, path, compare_runs.run_bench_path(run, 'cmoe'))), reference)
        for path, run in runs.values()]
    effective = [run.get('accounting', {}).get('effective_block_sparsity')
                 for _, run in runs.values()]
    effective = [value for value in effective if value is not None]
    return {
        'ppl': sum(ppl) / len(ppl),
        'acc': sum(one['macro']['acc'] for one in per_seed) / len(per_seed),
        'gold_nll': sum(one['macro']['gold_nll'] for one in per_seed) / len(per_seed),
        'effective': (sum(effective) / len(effective)) if effective else None,
        'n_seeds': len(runs),
    }


def main():
    reference = bench.load_samples(os.path.join(ROOT, DENSE_REFERENCE))
    dense = bench_stats.summarize(reference)['macro']
    dense_ppl = load(DENSE_PPL)['results'][PPL_DATASET]['ppl']

    for label, (suffix, template, n_active) in POINTS.items():
        moe_effective = moe_activated_sparsity(SHAPE, 8, n_active)
        payloads = {template.format(seed=seed): load(
            os.path.join(template.format(seed=seed), 'summary.json'))
            for seed in SEEDS}
        rows = []
        for width in WIDTHS:
            picked = {}
            for seed in SEEDS:
                path = template.format(seed=seed)
                found = compare_runs.pick(
                    {path: payloads[path]}, search_vector(suffix, width, seed), 'cmoe')
                picked.update(found)
            values = measure(picked, reference)
            values['effective'] = moe_effective
            rows.append((f'提案（beam 幅{width}）', values))
        for name, directory, pattern in PRUNE_ROWS:
            full = os.path.join(ROOT, directory, 'summary.json')
            if not os.path.exists(full):
                rows.append((name, None))
                continue
            picked = compare_runs.pick(
                {directory: load(os.path.join(directory, 'summary.json'))},
                pattern.format(value=VALUES[label]), 'cmoe')
            rows.append((name, measure(picked, reference)))

        print()
        print(f'### スパース率 {label}（校正はすべて slimpajama 16本×2048）')
        print()
        print('| 手法 | 実効ブロック削減率 | WikiText-2 PPL | acc | gold_nll |')
        print('|---|--:|--:|--:|--:|')
        for name, values in rows:
            if values is None:
                print(f'| {name} | （未測定） | | | |')
                continue
            effective = ('—' if values['effective'] is None
                         else f'{values["effective"]:.4f}')
            print(f'| {name} | {effective} | {values["ppl"]:.3f} | '
                  f'{values["acc"]:.4f} | {values["gold_nll"]:.3f} |')
        print(f'| dense（変換前） | 0.0000 | {dense_ppl:.3f} | {dense["acc"]:.4f} | '
              f'{dense["gold_nll"]:.3f} |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
