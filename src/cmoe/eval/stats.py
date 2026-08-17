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


def stratified_paired_bootstrap(differences_by_seed, reps=10000, seed=0):
    """seed で層化した、塊単位のブートストラップ。

    再抽出の単位は seed 内の評価塊であり、seed 自体は再抽出しない
    （seed は3本しかなく、そこから分布を推定はできない）。
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
    return {
        'mean_nll_difference': point,
        'lower': lower,
        'upper': upper,
        'confidence_level': 0.95,
        'repetitions': reps,
        'resampling_unit': 'paired-evaluation-chunk-within-seed',
        'seeds_resampled': False,
        'improved_seeds': sum(
            1 for values in differences_by_seed
            if sum(values) / len(values) < 0),
        'n_seeds': n_seeds,
    }
