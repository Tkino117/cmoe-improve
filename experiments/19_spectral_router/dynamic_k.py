"""同じ平均コストで、トークンごとに expert 数を変えたときの層出力誤差。

固定 Top-K は「どのトークンも同じ数の expert を要る」と決め打っている。要る量が
トークンで違うなら、平均を変えずに配り直すだけで誤差が下がるはずである。

ここで比べる配り方（どれも routed 全体の平均を Top-K に合わせる）:

* ``fixed``      … 現行。全トークンで K 個
* ``global``     … (トークン × expert) の score を全部並べ、上位 K·T 組を取る
* ``share``      … トークンごとに score を和で正規化してから、上位 K·T 組
* ``ratio``      … 最大 score に対する比が τ 以上のものを取る（τ は平均で決める）

score は ``cmoe``（現行ルーター）と ``oracle``（真の |h| 質量）の両方で見る。
オラクルの側で効かないなら、この軸自体に伸びしろが無いということになる。
"""

import argparse
import math
import os

import torch

from probe import DEFAULT_DIR, Material, batches, build_scorer, true_mass


def calibrate_threshold(material, scorer, budget, minimum, chunk=2048):
    """fit の上で、平均 expert 数が budget になる score のしきい値を1つ決める。

    ``global`` は (トークン × expert) を全部並べてから上位を取るので、推論では
    使えない — そのトークンだけを見て決められない。同じ配り方を1つのしきい値
    として取り出せば、トークンごとに独立に決まる形になる。しきい値は層ごとに
    1個のスカラーで、校正データから決まる。
    """
    values = []
    for x in batches(material.fit_z, chunk, material.device):
        scores = scorer(x)
        if minimum:
            # 最低配分ぶんは無条件に出るので、しきい値の対象から外す
            floor = scores.topk(minimum, dim=1).values[:, -1:]
            scores = scores.masked_fill(scores >= floor, -math.inf)
        values.append(scores.flatten().float())
    values = torch.cat(values)
    n_tokens = material.fit_z.shape[0]
    remaining = int((budget - minimum) * n_tokens)
    if remaining <= 0:
        return math.inf
    finite = values[values.isfinite()]
    if remaining >= finite.numel():
        return -math.inf
    return float(finite.topk(remaining).values[-1])


def allocate(scores, mode, budget, minimum=0, threshold=None):
    """[tokens, experts] の score から 0/1 の選択行列を作る。合計は budget。"""
    n_tokens, n_experts = scores.shape
    total = budget * n_tokens
    if mode == 'fixed':
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, scores.topk(budget, dim=1).indices, True)
        return mask
    if mode == 'share':
        values = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    elif mode == 'ratio':
        values = scores / scores.amax(dim=1, keepdim=True).clamp_min(1e-12)
    elif mode == 'global':
        values = scores
    elif mode == 'tau':
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if minimum:
            mask.scatter_(1, scores.topk(minimum, dim=1).indices, True)
        return mask | (scores >= threshold)
    else:
        raise ValueError(mode)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if minimum:
        mask.scatter_(1, scores.topk(minimum, dim=1).indices, True)
    remaining = total - int(mask.sum())
    if remaining > 0:
        free = values.masked_fill(mask, -math.inf).flatten()
        mask.flatten()[free.topk(remaining).indices] = True
    return mask


@torch.no_grad()
def evaluate(material, names, modes, minimum, chunk=2048):
    fit = material.fit_z
    scorers = {name: build_scorer(name, material, fit)[0] for name in names}
    budget = material.topk
    thresholds = (
        {name: calibrate_threshold(material, scorer, budget, minimum)
         for name, scorer in scorers.items()} if 'tau' in modes else {})
    stats = {(name, mode): {'mass': 0.0, 'sq': 0.0, 'k': 0}
             for name in names for mode in modes}
    total_mass = 0.0
    output_norm = 0.0
    n_tokens = 0

    for x in batches(material.validation_z, chunk, material.device):
        n_tokens += x.shape[0]
        mass = true_mass(material, x)
        outputs = []
        for index in range(len(material.groups)):
            if not material.groups[index].numel():
                outputs.append(torch.zeros_like(x))
                continue
            _, _, down = material.expert_rows(index)
            outputs.append(material.h(x, index) @ down.T)
        dense = sum(outputs)
        shared = (material.h(x, 0).abs_().sum(dim=1)
                  if material.groups[0].numel()
                  else torch.zeros(x.shape[0], device=x.device))
        total_mass += float((shared + mass.sum(dim=1)).sum(dtype=torch.float64))
        output_norm += float(dense.square().sum(dtype=torch.float64))
        routed = torch.stack(outputs[1:], dim=1)

        for name, scorer in scorers.items():
            scores = scorer(x)
            for mode in modes:
                mask = allocate(scores, mode, budget, minimum,
                                thresholds.get(name))
                state = stats[(name, mode)]
                state['k'] += int(mask.sum())
                state['mass'] += float(
                    (shared + (mass * mask).sum(dim=1)).sum(dtype=torch.float64))
                selected = (routed * mask[:, :, None]).sum(dim=1)
                error = dense - (outputs[0] + selected)
                state['sq'] += float(error.square().sum(dtype=torch.float64))
        del outputs, dense, routed, mass

    rows = {}
    for key, state in stats.items():
        rows[key] = {
            'mass': state['mass'] / total_mass,
            'err': math.sqrt(state['sq'] / output_norm),
            'mean_k': state['k'] / n_tokens,
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', default=DEFAULT_DIR)
    parser.add_argument('--scorers', default='cmoe,oracle')
    parser.add_argument('--modes', default='fixed,global,share,ratio')
    parser.add_argument('--minimum', type=int, default=0,
                        help='どのトークンにも最低これだけ配る')
    parser.add_argument('--layers', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    names = [f.strip() for f in args.scorers.split(',') if f.strip()]
    modes = [f.strip() for f in args.modes.split(',') if f.strip()]
    paths = sorted(os.path.join(args.dump, e) for e in os.listdir(args.dump)
                   if e.endswith('.pt'))
    if args.layers:
        wanted = {int(f) for f in args.layers.split(',')}
        paths = [p for p in paths if int(os.path.basename(p)[5:7]) in wanted]

    device = torch.device(args.device)
    per_layer = []
    for path in paths:
        material = Material(torch.load(path, map_location='cpu'), device)
        per_layer.append(evaluate(material, names, modes, args.minimum))
        print(f'層 {material.layer} 済', flush=True)
        del material
        torch.cuda.empty_cache()

    print(f'\n=== 層平均（最低配分 {args.minimum}）===')
    print(f'{"scorer":<12} {"mode":<8} {"mean_k":>7} {"mass":>9} {"err":>8}')
    for name in names:
        for mode in modes:
            values = [rows[(name, mode)] for rows in per_layer]
            print(f'{name:<12} {mode:<8} '
                  f'{sum(v["mean_k"] for v in values)/len(values):>7.3f} '
                  f'{sum(v["mass"] for v in values)/len(values):>9.6f} '
                  f'{sum(v["err"] for v in values)/len(values):>8.5f}')


if __name__ == '__main__':
    main()
