"""対応のある比較の統計。

このプロジェクトで問題になる差は PPL で 0.005〜0.07 と小さく、seed 間の揺れ
（±0.02〜0.04）と同じ桁である。したがって「平均がどちらが小さいか」だけでは
足りず、塊ごとの対応のある差と、その信頼区間まで見る。

CMoE-ref の各ドライバに散っていた ``paired_differences`` /
``stratified_paired_bootstrap`` の移送。
"""

import torch


def paired_differences(baseline, candidate):
    """同じ塊どうしの平均 NLL の差（負なら candidate が良い）。"""
    left = baseline.chunk_mean_nlls
    right = candidate.chunk_mean_nlls
    if len(left) != len(right):
        raise ValueError(
            f'対応が取れない: 塊が {len(left)} 個と {len(right)} 個')
    return [after - before for before, after in zip(left, right)]


def stratified_paired_bootstrap(differences_by_seed, reps=10000, seed=0,
                                key='mean_nll_difference',
                                unit='paired-evaluation-chunk-within-seed',
                                lower_is_better=True):
    """seed で層化した、塊単位のブートストラップ。

    再抽出の単位は seed 内の評価塊であり、seed 自体は再抽出しない
    （seed は3本しかなく、そこから分布を推定はできない）。

    層の中身は PPL では「1 seed 分の評価塊」だが、選択問題では「1 seed × 1
    タスク分の問題」になる。層をどう切るかは呼び出し側の問題で、ここは渡された
    層を等しく重み付けして平均するだけである（タスクを層にすれば、問題数が
    一桁違うタスクどうしでもマクロ平均になる）。``key`` / ``unit`` /
    ``lower_is_better`` は、何を測った差なのかを結果に書き残すためにある。
    """
    if not differences_by_seed:
        raise ValueError('差の列が空')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    boot = torch.zeros(reps, dtype=torch.float64)
    point = 0.0
    n_seeds = len(differences_by_seed)
    for values in differences_by_seed:
        tensor = torch.tensor(values, dtype=torch.float64)
        point += float(tensor.mean()) / n_seeds
        indices = torch.randint(
            tensor.numel(), (reps, tensor.numel()), generator=generator)
        boot.add_(tensor[indices].mean(dim=1) / n_seeds)
    lower, upper = torch.quantile(
        boot, torch.tensor([0.025, 0.975], dtype=boot.dtype)).tolist()
    better = ((lambda mean: mean < 0) if lower_is_better else (lambda mean: mean > 0))
    return {
        key: point,
        'lower': lower,
        'upper': upper,
        'confidence_level': 0.95,
        'repetitions': reps,
        'resampling_unit': unit,
        'seeds_resampled': False,
        'improved_seeds': sum(
            1 for values in differences_by_seed
            if better(sum(values) / len(values))),
        'n_seeds': n_seeds,
    }
