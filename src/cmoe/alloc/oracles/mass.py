"""R(x): その層で、dense FFN の活性質量のうち何割を MoE がまだ実行するか。

CMoE-ref の ``xsearch/simulate.py`` の移送。トークン t で MoE が走らせるのは
shared のニューロンと、ルーターが選んだ routed expert のニューロンだけで、
走らせなかったぶんがそのまま誤差になる。CMoE 自身がニューロンを |h| で採点して
いるので、候補の良さは回収した L1 質量の割合で測る:

    R(x) = Σ_t Σ_{i ∈ E_t(x)} |H_ti| / Σ_t Σ_i |H_ti|

``squared=True`` は各ニューロンを H^2 で重み付ける。真の出力誤差 L(x) の対角
近似で、スパイクが二乗で効き、ゼロ付近の大多数が消える（w_down の大きさと
ニューロン間の打ち消しは無視されたまま）。

破ってはいけない規則が2つある。

* 質量は**真の H**（素の重み × 素の入力）から取る。プロファイル用の正規化した
  h から取らない
* 集計は fp32 で行う。bf16 の和は候補の順序が入れ替わるほど粗く、順序を付ける
  ことがこの数の唯一の仕事である

スコアは「取りこぼした質量の割合」 1 - R で、小さいほど良い。接頭辞のスコアは
層ごとの取りこぼしの**和**にする。層ローカル指標は層ごとに独立に定義されて
いるので、これで同じ深さの接頭辞どうしが比べられる。
"""

import torch

from cmoe.alloc.base import ScoreResult
from cmoe.alloc.oracles.base import PrefixOracleBase, apply_weights


def group_mass(h, groups, squared=False):
    """ニューロン群ごとの、トークンごとの |h| ないし h^2 質量。fp32。

    R(x)・層ローカル誤差・ルーター診断が同じ「質量」を指すための1本。
    """
    out = torch.zeros(h.shape[0], len(groups), dtype=torch.float32,
                      device=h.device)
    for index, group in enumerate(groups):
        if len(group) == 0:      # x=0 の shared expert
            continue
        selected = torch.as_tensor(list(group), dtype=torch.long, device=h.device)
        if squared:
            # 二乗は fp32 に上げてから。bf16 のまま二乗すると相対誤差が倍になり、
            # それはこの指標が重く見たいスパイクのところで起きる
            out[:, index] = h.index_select(1, selected).to(
                torch.float32).square_().sum(dim=1)
        else:
            out[:, index] = h.index_select(1, selected).abs_().sum(
                dim=1, dtype=torch.float32)
    return out


def neuron_to_expert(partition, n_neurons, device):
    """ニューロン → それを走らせる routed expert の番号 [n_neurons]。

    shared のニューロンは routed の1つ先の番号（番兵）へ送る。トークンごとの
    選択をこれで引くと [トークン, ニューロン] の「走らせた印」が一度に作れる。

    分割がニューロンをちょうど1回ずつ覆っていることも、ここで検査する。数だけを
    数えると、重複したニューロンが抜けたニューロンを埋め合わせてしまい、抜けた
    ぶんの質量が分子からも分母からも黙って消える。
    """
    n_routed = len(partition.routed_groups)
    group_of = torch.full((n_neurons,), -1, dtype=torch.long, device=device)
    entries = 0
    for index, group in enumerate(partition.routed_groups):
        if len(group) == 0:
            continue
        group_of[torch.as_tensor(list(group), dtype=torch.long, device=device)] = index
        entries += len(group)
    shared = partition.shared_group
    if len(shared):
        group_of[torch.as_tensor(list(shared), dtype=torch.long, device=device)] = n_routed
        entries += len(shared)
    if entries != n_neurons or int((group_of < 0).sum()):
        raise ValueError(
            f'分割が {n_neurons} 個のニューロンをちょうど1回ずつ覆っていない '
            f'（延べ {entries} 個、割り当てられたのは '
            f'{int((group_of >= 0).sum())} 個）')
    return group_of


@torch.no_grad()
def router_selection(moe, z, token_chunk=None):
    """その層に載る当のルーターで、トークンごとの選択を取る。

    ``MoE.forward`` が自分の入力にするのと同じ形（1トークン1行）に畳んでから、
    同じモジュールを呼ぶ。層に載る物と測る物が同じ1個なので、「offline で再現
    した選択が実機と合っているか」という問いがそもそも立たない。

    バッチを分けて進めているとき z はホストに残り、ルーターの重みは層と同じ
    デバイスにある。塊ごとに重みの側へ渡し、選択は z と同じ側へ戻す — 選択は
    このあと z と同じ側にある H と突き合わせるためのものだからである。
    """
    if z.shape[-1] != moe.gate.dim:
        raise ValueError(f'z の最終次元が {z.shape[-1]}、ルーターは {moe.gate.dim}')
    router = moe.gate
    flat = z.reshape(-1, router.dim)
    device = router.classifier.weight.device
    step = token_chunk or flat.shape[0]
    rows = []
    for start in range(0, flat.shape[0], step):
        _, indices = router(flat[start:start + step].to(device))
        rows.append(indices.to(flat.device))
    return torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]


def selected_mask(routed_mass, indices):
    """選ばれた routed expert に 1 を立てた [トークン, routed] のマスク。

    gather ではなくマスクで足すのは、回収の和が全体の和と**同じ順序で同じ列**を
    足すようにするためである。全 expert を選んだときにちょうど 1.0 になり、
    1 ulp ずれた 1.0 にならない。
    """
    mask = torch.zeros_like(routed_mass)
    if indices.numel():
        mask.scatter_(1, indices.to(torch.int64), 1.0)
    return mask


@torch.no_grad()
def recovered_mass(h, partition, indices, squared=False, token_chunk=None,
                   device=None, weights=None):
    """(R, 内訳)。h は [トークン, ニューロン] の真の H。

    ``weights`` を渡すと、位置ごとの質量にその重みを掛けてから比を取る（分子も
    分母も同じ重みで、割合であることは保たれる）。渡さなければ全位置が等しい。

    バッチを分けているとき h はホストに残る。層ローカル誤差と同じく、計算は
    重みが載っている側で塊ごとに回し、トークンごとの和だけを戻す。和はトークン
    ごとに閉じているので、分けても値は変わらない。
    """
    n_tokens = h.shape[0]
    step = token_chunk or n_tokens
    shared_rows, selected_rows, total_rows = [], [], []
    for start in range(0, n_tokens, step):
        stop = min(start + step, n_tokens)
        chunk = h[start:stop].to(device) if device is not None else h[start:stop]
        rows = (indices[start:stop].to(device) if device is not None
                else indices[start:stop])
        shared_chunk = group_mass(chunk, [partition.shared_group], squared)[:, 0]
        routed = group_mass(chunk, partition.routed_groups, squared)
        selected_chunk = (routed * selected_mask(routed, rows)).sum(dim=1)
        shared_rows.append(shared_chunk.to(h.device))
        selected_rows.append(selected_chunk.to(h.device))
        total_rows.append((shared_chunk + routed.sum(dim=1)).to(h.device))
        del chunk, rows, routed, shared_chunk, selected_chunk

    shared = torch.cat(shared_rows) if len(shared_rows) > 1 else shared_rows[0]
    selected = torch.cat(selected_rows) if len(selected_rows) > 1 else selected_rows[0]
    per_token_total = torch.cat(total_rows) if len(total_rows) > 1 else total_rows[0]
    per_token_recovered = shared + selected
    # 重みは最後に1回だけ掛ける。塊ごとの和はトークンで閉じているので、掛ける
    # 場所を後ろへ寄せても値は変わらない
    shared = apply_weights(shared, weights)
    selected = apply_weights(selected, weights)
    per_token_recovered = apply_weights(per_token_recovered, weights)
    per_token_total = apply_weights(per_token_total, weights)
    total = float(per_token_total.sum())
    # `not (x > 0)` と書くのは、NaN があらゆる比較に失敗するからである。NaN の
    # まま通すと、候補の大小比較が当たり外れで決まる
    if not total > 0:
        raise ValueError(f'活性質量の合計が {total}。H が空か非有限')
    r = float(per_token_recovered.sum() / per_token_total.sum())
    if not 0.0 <= r <= 1.0:
        raise ValueError(f'R={r} は割合になっていない。H がおそらく非有限')
    return r, {
        'shared_mass': float(shared.sum()),
        'selected_routed_mass': float(selected.sum()),
        'total_mass': total,
        'n_tokens': int(h.shape[0] if weights is None else (weights > 0).sum()),
    }


class MassOracle(PrefixOracleBase):
    """層ローカル: 取りこぼした活性質量の割合。いちばん安い採点。

    測るのに要るのはこの層の H とルーターの選択だけで、後続の層を1つも走らせ
    ない。``squared=True`` にすると H^2 で重み付ける（出力誤差の対角近似）。
    """

    name = 'mass'
    # 後続層を走らせないので、1回ぶんのコストはこの層の FFN 1回に相当する
    cost_unit = 'layer_ffn_evals'
    # 走らせなかった質量を測るので、A >= N では何も測らない
    needs_routing = True

    def __init__(self, walk, squared=False, token_chunk=None):
        super().__init__(walk)
        self.squared = squared
        self.token_chunk = token_chunk

    @torch.no_grad()
    def measure(self, profile, carved, state, child):
        h = self.walk.true_activations(profile)
        # 集計は重みが載っている側で走らせ、トークンの塊だけを行き来させる
        # （層ローカル誤差と同じ扱い）
        device = profile.dense.down_proj.weight.device
        # 覆いの検査のためだけに引く。重複や抜けがあると、そのぶんの質量が
        # 分子からも分母からも黙って消える
        neuron_to_expert(carved.partition, h.shape[1], device)
        indices = router_selection(carved.moe, profile.z, self.token_chunk)
        if indices.shape[0] != h.shape[0]:
            raise ValueError(
                f'ルーターが {indices.shape[0]} 行を返した（トークンは '
                f'{h.shape[0]} 個）')
        r, details = recovered_mass(h, carved.partition, indices, self.squared,
                                    self.token_chunk, device,
                                    self.walk.flat_score_weights())
        details.update({'r': r, 'missed': 1.0 - r, 'layer': profile.layer,
                        'x': carved.n_shared, 'topk': carved.topk,
                        'squared': self.squared})
        return ScoreResult(score=state.score + (1.0 - r), cost=1.0,
                           cost_unit=self.cost_unit, details=details)
