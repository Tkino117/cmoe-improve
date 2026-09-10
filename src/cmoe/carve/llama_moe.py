"""[対照] LLaMA-MoE / LLaMA-MoE-v2 の分割規則。**どちらも提案手法ではない。**

CMoE 最新版 (ACL 2026) Table 1 が並べている MoE Restructuring の比較手法である。

**移すのは分割（軸3）だけである。** LLaMA-MoE は継続事前学習 200B トークン、
LLaMA-MoE-v2 は2段の post-training 約7B トークンを前提にしており、分割直後の
状態は原典の評価対象ではない。CMoE 論文も同じ扱いをしていて、§5.1 は
"All baselines are re-implemented by us and fine-tuned with LoRA under matched
data budget"、Table 6 の脚注は "† Split-only; training time not included" と
書いている。ここは**追加学習をしない土俵**（report/25 と同じ）なので、
`ew_rule.py` を「EW-rule」と呼んだのと同じく、**手法そのものの比較ではない**。

移送元は公式実装:

* v1 … `pjlab-sys4nlp/llama-moe` の `smoe/utils/expert_construction/expert_split.py`
        の `RandomSplit`
* v2 … `OpenSparseLLMs/LLaMA-MoE-v2` の
        `smoe/utils/expert_construction/expert_split_residual.py` の
        `GradientSplitResidual`（`criterion="max"`, `share_neurons=False`）と
        `smoe/entrypoint/expert_construction/split/split_gradient_get_grads_v2.py`

## v1（Random）

``labels = [0..E-1] * (i // E)`` を並べて shuffle するだけ。**等サイズ・非重複の
一様ランダム割り当て**である。原典は shared expert を持たないので、動作点は
``--alloc uniform0``（x=0、全 expert が routed）が対応する。原典が
Clustering / Co-activation Graph / Gradient も試したうえで **Random を最良と
結論している**ので、ここも Random だけを移す。

## v2（Gradient + Residual）

3段。1 と 2 は層ごと・クラスタごとの重要度を作る**プローブ**で、
`experiments/28_llama_moe/probe.py` が担う。ここが受け取るのはその出力である。

1. **ゲート（層ごと）** … MLP 入力の隠れ状態を、routed expert と同数の
   クラスタへ balanced k-means で分ける。中心が原典の gate 重みになる
2. **クラスタごとの重要度** … LM 損失を backward し、各トークンを
   ``argmax(h · center^T)`` でクラスタへ割り当て、中間活性 f について
   ``|f ⊙ ∇_f L|`` をクラスタ内で平均する。層 ℓ・クラスタ c ごとに
   [intermediate] のベクトルが1本出る
3. **分割**（この module）… residual（= shared）に「多くのクラスタが上位に
   挙げたニューロン」を詰め、残りを貪欲な最大値選択で routed へ等サイズに配る

x（shared の数）は原典の ``expert_num_residual`` にそのまま対応する。原典の
主構成 ``1+7top1`` は x=1 である。**クラスタ数は routed expert 数** なので、
x を変えるとプローブを取り直す必要がある。
"""

import itertools
import random

import numpy as np
import torch

from cmoe.carve.base import Partition


def _representatives(groups, features):
    """クラスタ重心に最も近いニューロンを代表に取る。

    現行 CMoE と同じ規則である（``carve/cmoe.py`` の3段目）。ルーターは代表の
    行から作られるので、ここを変えるとルーター軸も動いてしまう。
    """
    representatives = []
    for group in groups:
        columns = features[:, group]
        centre = columns.mean(dim=1, keepdim=True)
        distance = (columns - centre).abs().sum(dim=0)
        representatives.append(int(group[int(torch.argmin(distance))]))
    return tuple(representatives)


class RandomSplitCarver:
    """[対照] LLaMA-MoE v1 の Random 分割。

    ニューロンを等サイズ・非重複でランダムに割る。**校正データを一切読まない。**
    shared expert は「割り当てのうち先頭 x 個ぶんの束」をそのまま使う（原典に
    shared は無いので、x>0 で走らせるのはこちらの拡張であり、その旨を表に書く）。

    seed は校正の seed と同じものを与える。分割そのものが確率的な唯一の方式
    なので、seed 間のばらつきがそのまま手法のばらつきになる。
    """

    name = 'llama_moe_random'
    k_act = 10

    def __init__(self, n_experts, seed=0):
        if n_experts < 1:
            raise ValueError(f'n_experts は 1 以上 (受け取った値: {n_experts})')
        self.n_experts = n_experts
        self.seed = seed

    @torch.no_grad()
    def carve(self, dense, rates, markers, n_shared, z=None, layer=None):
        if not 0 <= n_shared < self.n_experts:
            raise ValueError(
                f'n_shared={n_shared} は 0..{self.n_experts - 1} の外')
        if dense.intermediate_size % self.n_experts:
            raise ValueError(
                f'{dense.intermediate_size} ニューロンは {self.n_experts} で'
                '割り切れない; 余りが黙って捨てられる')
        size = dense.intermediate_size // self.n_experts
        # 原典 RandomSplit: [0..E-1] を size 回並べて shuffle する
        labels = list(range(self.n_experts)) * size
        # 層ごとに違う並びにする。同じ seed で層をまたいで同じ割り当てを使うと、
        # 「ランダム分割」ではなく「1つの固定分割」を測ることになる
        # 種は文字列で作る。``random.Random`` は文字列を sha512 で消費するので、
        # プロセスや PYTHONHASHSEED に依存しない（data/slimpajama.py と同じ）
        random.Random(f'{self.seed}:{layer}').shuffle(labels)
        buckets = [[] for _ in range(self.n_experts)]
        for index, label in enumerate(labels):
            buckets[label].append(index)

        shared = tuple(itertools.chain(*buckets[:n_shared])) if n_shared else ()
        routed = [tuple(bucket) for bucket in buckets[n_shared:]]
        partition = Partition(
            n_experts=self.n_experts, n_shared=n_shared,
            expert_groups=(shared,) + tuple(routed),
            representative_indices=_representatives(routed, markers))
        return partition.check(dense.intermediate_size)


class LlamaMoEV2Carver:
    """[対照] LLaMA-MoE-v2 の Gradient + Residual 分割（MLP 側のみ）。

    層ごと・クラスタごとの重要度をプローブから受け取る。``scores[layer]`` は
    [クラスタ, intermediate] で、クラスタ数は routed expert 数に等しい。

    **Attention MoE は移していない。** 原典の主表の構成も ``MLP-MoE (8top2)`` /
    ``MLP-MoE (1+7top1)`` であり、attention と併用したモデルは主結果に出て
    こない。落としたのは原典が主結果に使っていない部分である。
    """

    name = 'llama_moe_v2'
    k_act = 10

    def __init__(self, n_experts, scores=None):
        if n_experts < 1:
            raise ValueError(f'n_experts は 1 以上 (受け取った値: {n_experts})')
        self.n_experts = n_experts
        # {層番号: [クラスタ, intermediate] のテンソル}
        self.scores = scores or {}

    def layer_scores(self, layer, n_routed, intermediate_size):
        if layer is None:
            raise ValueError(
                'llama_moe_v2 は層ごとの重要度を読むので、層番号が要る。'
                '探索（cmoe search）の経路からは使えない')
        try:
            table = self.scores[layer]
        except KeyError:
            raise ValueError(
                f'層 {layer} の重要度がプローブの出力に無い') from None
        table = torch.as_tensor(table, dtype=torch.float32)
        if tuple(table.shape) != (n_routed, intermediate_size):
            raise ValueError(
                f'層 {layer} の重要度は {tuple(table.shape)} だが、'
                f'routed {n_routed} × {intermediate_size} が要る。'
                'x を変えたらプローブを取り直す（クラスタ数 = routed 数）')
        return table

    @torch.no_grad()
    def carve(self, dense, rates, markers, n_shared, z=None, layer=None):
        if not 0 <= n_shared < self.n_experts:
            raise ValueError(
                f'n_shared={n_shared} は 0..{self.n_experts - 1} の外')
        n_routed = self.n_experts - n_shared
        size = dense.intermediate_size // self.n_experts
        table = self.layer_scores(layer, n_routed, dense.intermediate_size)

        if dense.intermediate_size % self.n_experts:
            raise ValueError(
                f'{dense.intermediate_size} ニューロンは {self.n_experts} で'
                '割り切れない; 余りが黙って捨てられる')

        order = [torch.argsort(row, descending=True).tolist() for row in table]
        residual = _residual_neurons(order, n_shared * size, size, n_routed)
        routed = _greedy_assign(table, order, residual, n_routed, size)

        groups = (tuple(sorted(residual)),) + tuple(
            tuple(sorted(group)) for group in routed)
        partition = Partition(
            n_experts=self.n_experts, n_shared=n_shared,
            expert_groups=groups,
            representative_indices=_representatives(groups[1:], markers))
        return partition.check(dense.intermediate_size)


def _residual_neurons(order, need, size, n_routed):
    """residual に「多くのクラスタが上位に挙げたニューロン」を詰める。

    原典 ``split_with_neuron_sharing`` の移送。各クラスタの上位 ``size`` 個を
    見て、**選んだクラスタ数が多いニューロンから**順に residual へ入れる。
    入れたぶんを取り除いて、埋まるまで繰り返す。
    """
    if need == 0:
        return []
    picked, taken = [], set()
    remaining = [list(row) for row in order]
    while len(picked) < need:
        counts = {}
        for row in remaining:
            for index in row[:size]:
                counts[index] = counts.get(index, 0) + 1
        if not counts:
            raise ValueError('residual を埋めるニューロンが尽きた')
        for repeats in range(n_routed, 0, -1):
            found = sorted(index for index, count in counts.items()
                           if count == repeats)
            if not found:
                continue
            found = found[:need - len(picked)]
            picked.extend(found)
            taken.update(found)
            break
        else:
            raise ValueError('residual を埋めるニューロンが尽きた')
        remaining = [[index for index in row if index not in taken]
                     for row in remaining]
    return picked


def _greedy_assign(table, order, residual, n_routed, size):
    """残りを貪欲な最大値選択で routed へ等サイズに配る。

    原典 ``split_without_neuron_sharing`` の移送。各クラスタの「まだ取られて
    いない最上位」を見比べ、**重要度そのものが最大**のものを取る、を繰り返す。
    同点のときに原典は ``>=`` で後ろのクラスタを採るので、そこも合わせてある。
    """
    values = table.tolist()
    used = set(residual)
    cursors = [0] * n_routed
    counts = [0] * n_routed
    groups = [[] for _ in range(n_routed)]
    total = n_routed * size
    placed = 0
    while placed < total:
        best_score, best_expert, best_neuron = None, -1, -1
        for expert in range(n_routed):
            row = order[expert]
            while cursors[expert] < len(row) and row[cursors[expert]] in used:
                cursors[expert] += 1
            if counts[expert] == size or cursors[expert] >= len(row):
                continue
            neuron = row[cursors[expert]]
            score = values[expert][neuron]
            if best_score is None or score >= best_score:
                best_score, best_expert, best_neuron = score, expert, neuron
        if best_expert < 0:
            raise ValueError('routed を埋めるニューロンが尽きた')
        groups[best_expert].append(best_neuron)
        used.add(best_neuron)
        cursors[best_expert] += 1
        counts[best_expert] += 1
        placed += 1
    return groups
