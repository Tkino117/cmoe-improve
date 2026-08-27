"""方式5: 代表は凍結したまま、expert ごとの score の gain と offset を探す。

方式1〜4 が動かすのはどれも同じ対象 — expert を代表するニューロン — だった。
その選択は**方向**を決め、ルーターはその方向が出す score で expert を比べる。
どの探索にも入っていない自由度が2つある。

* router 行の L2 正規化は、どの expert の利得もちょうど1に固定する。方向は
  合っているのに score が系統的に小さい代表を、大きくする手段が無い
* ``Router.extra_bias`` — 配備済みのルーターが softmax の後に足す expert ごとの
  offset — は、学習なしの経路では常にゼロのままである

Top-K は routed score の**順序しか読まない**ので、expert ごとの gain と offset は
選択を変えながら、ルーターの演算・パラメータ数・推論コストのどれも変えない。
この方式は渡された代表を凍結し、その 2N 個の数を方式4 と同じ目的関数

    R = Σ_t (M_shared(t) + Σ_{j∈TopK(t)} M_j(t)) / Σ_t (M_shared(t) + Σ_j M_j(t))

に合わせる。

1座標が厳密かつ安価なのは方式4 と同じ理由による。他の expert を固定し、その
最終 score の K 番目に大きい値を b_K(t) とすると、expert e が Top-K に入る条件は
自分の最終 score がそれを超えることだけである:

    gain    α_e · s_e(t) > b_K(t)  ⟺  α_e > b_K(t) / s_e(t)
    offset  p_e(t) + β_e  > b_K(t)  ⟺  β_e  > b_K(t) − p_e(t)

どちらも、1つのスカラーについての目的関数が階段関数
``const + Σ_t 1[τ(t) < knob] · delta(t)`` になる。トークンごとの閾値を並べ替えて
累積 delta の最良接頭辞を取れば、それが**連続体全体の厳密な最適応答**である
— 1座標につきソート1回で、格子は要らない。

gain は offset をゼロにしたまま先に合わせる。そこでは softmax が生の score の
順序を保つので無視できる。offset はそのあと、凍結した gain に対して合わせる
（softmax は1回計算してそのまま）。gain へ戻る3段目は**厳密ではない** — offset が
1つでも非ゼロになると softmax の分母が全 expert を結合するので、縮約がもう
成り立たない。だから3段目は「まだ厳密なふり」をせずに止める。結果は
「offset ゼロでの各 gain について座標ごとに最適」かつ「その gain での各 offset に
ついて座標ごとに最適」であって、両者について同時に最適ではない。3段目が拾う
はずのものは意図的に置いていく。

gain を ``[1/gain_limit, gain_limit]`` の中で探すのは、上の順序の議論自体に限界が
あるからである。配備の softmax は fp32 で走るので、1つの expert の score が十分
遠くまで走ると他の指数が underflow し、確率が一律 0.0 に潰れ、Top-K が score では
なく**添字**で同点を割り始める。そこから先では、縮約はルーターがしない選択を
予測する。この境界は自由な選択ではなく明示的な停止点である — それを越えた gain は
より小さい gain でも offset でも表現できない選択に届くが、そこは単に探索しない。
既定を正当化するのは証明ではなく測定である: 256 まで広げると合わせた gain は
~200 まで動き、fit・held-out のどちらの回収率も良くならなかった。飽和は gain だけ
ではなくその層の生の score がもともとどれだけ離れているかに依るからである。
だから前提は仮定ではなく記録する — ``order_faithful`` と
``order_mismatch_fraction`` が、配備の Top-K が fit トークン上で生の score の順序と
まだ一致しているかを層ごとに報告する。Llama-2-7B では32層のうち3層で一致しない。

fit を配備に対して正直に保つ規則が2つある。どの試行も gain をモデル自身の dtype で
router 行へ実体化し、そこから score 行列を**丸ごと**作り直す — 決して1列ずつでは
ない。1行の GEMM は、ルーターの N 行の呼び出しと同じ順序では hidden 次元を縮約
しないので、差し替えた1列と隣の列を比べることは別の演算どうしを比べることになる。
（ルーターとのビット一致はバッチの形ごとには成り立つが、形をまたいでは成り立た
ない。cuBLAS は 4096 行の呼び出しと 8192 行の呼び出しを別に縮約する。探索が要る
のは「1回の比較に出てくる expert が全部1回の呼び出しから出た」という、より弱い
性質である。）そして両段のどの手も、配備の選択経路（softmax・offset・Top-K）を
通して測り直し、R が本当に増えたときだけ残す。単調性が仮定ではなく構成で出る。

CMoE-ref の ``routerlab/methods/score_calibration.py`` の移送。
"""

import copy
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

# 質量と score は方式4 から import する（作り直さない）。2つの方式が比べられる
# のは、同一の目的関数を同一の質量の上で採点しているからであって、その規則の
# 2つ目の写しは、黙ってずれる2つ目の場所になる。
from cmoe.router.methods.oracle_correlation import (activation, flatten_z,
                                                    validate_weights)
from cmoe.router.methods.oracle_recovery import clean_group, expert_masses

# gain の既定の上限。配備ルーターの fp32 softmax が生の score の順序を保ち続ける
# ところに置く（モジュールの docstring 参照）。定数ではなく引数なのは、安全な幅が
# 「その層の score がもともとどれだけ離れているか」というモデルの性質に依るから
# である。offset にはこの種の上限が要らず、勝った区間が片側に開いているときは
# 中点が取れないので1つ分だけ進める（softmax の出力が張れる幅そのもの）。
GAIN_LIMIT = 16.0
BIAS_BOUNDARY_MARGIN = 1.0


@dataclass(frozen=True)
class CalibrationSelection:
    """合わせたつまみ、その回収率、そこへ至った経路。"""

    representatives: tuple
    gains: tuple
    biases: tuple
    recovery: float
    initial_recovery: float
    gain_recovery: float
    moves: tuple
    sweeps_used: dict
    terminations: dict
    gain_limit: float
    faithfulness: dict
    n_tokens: int

    def metadata(self):
        return {
            'representatives': list(self.representatives),
            'gains': list(self.gains),
            'biases': list(self.biases),
            'fit_recovery': self.recovery,
            'initial_recovery': self.initial_recovery,
            'gain_recovery': self.gain_recovery,
            'accepted_moves': [dict(move) for move in self.moves],
            'sweeps_used': dict(self.sweeps_used),
            'terminations': dict(self.terminations),
            'gain_limit': self.gain_limit,
            **self.faithfulness,
            'n_tokens': self.n_tokens,
        }


def unit_rows(gate_weight, up_weight, representatives, router_normalized):
    """gain=1 が意味する router 行、つまり方式1〜4 が配備するものそのもの。"""
    indices = list(representatives)
    gate = gate_weight[indices]
    up = up_weight[indices]
    if router_normalized:
        gate = F.normalize(gate, p=2, dim=1)
        up = F.normalize(up, p=2, dim=1)
    return gate.contiguous(), up.contiguous()


def scaled_up_rows(unit_up, gains):
    """gain を classifier 行へ、モデル自身の dtype で畳み込む。

    探索と、返すルーターは、行を同じ作り方で実体化しなければならない。でないと
    fit は出荷されない重みを採点したことになる。
    """
    rows = unit_up.clone()
    for expert, gain in enumerate(gains):
        rows[expert] = unit_up[expert] * gain
    return rows


@torch.no_grad()
def score_columns(z, gate_rows, up_rows, chunk_size):
    """全 fit トークンに対する、各 expert の配備 softmax 前 score。"""
    scores = torch.empty((z.shape[0], gate_rows.shape[0]),
                         dtype=gate_rows.dtype, device=gate_rows.device)
    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        chunk = z[start:stop].to(gate_rows.device)
        scores[start:stop] = activation(chunk, gate_rows, up_rows).abs_()
    return scores


def deployed_final(scores, biases):
    """``Router.forward`` が順位を付ける当の量: fp32 の softmax、そのあと offset。"""
    return scores.softmax(dim=-1, dtype=torch.float32) + biases


def order_faithfulness(scores, topk):
    """fp32 softmax は、生の score が付けた順位をまだ保っているか。

    これが gain の段を厳密にしている前提である。仮定ではなく記録するのは、
    黙って壊れるからである — 指数が underflow すると Top-K は添字で同点を割り、
    動く質量は縮約が勘定した質量ではなくなる。判定に割合と score の広がりを
    添えるのは、「100万トークンに1つ」と「層の半分」が別の発見であり、bool では
    その2つを区別できないからである。
    """
    spread = scores.float()
    spread = float((spread.amax(dim=1) - spread.amin(dim=1)).max())
    if topk == 0:
        return {'order_faithful': True, 'order_mismatch_fraction': 0.0,
                'max_score_spread': spread}
    deployed = deployed_final(scores, 0.0).topk(topk, dim=1).indices
    raw = scores.float().topk(topk, dim=1).indices
    mismatch = (deployed.sort(dim=1).values != raw.sort(dim=1).values).any(dim=1)
    return {
        'order_faithful': not bool(mismatch.any()),
        'order_mismatch_fraction': float(mismatch.to(torch.float32).mean()),
        'max_score_spread': spread,
    }


def _recovery(final, routed, shared_mass, total, topk):
    indices = final.topk(topk, dim=1).indices
    selected = float(routed.gather(1, indices).sum(dtype=torch.float64))
    return (shared_mass + selected) / total


def _threshold_and_delta(final, routed, expert, topk):
    """他の expert の K 番目に大きい score と、それと入れ替わる質量。"""
    others = [index for index in range(final.shape[1]) if index != expert]
    picked = final[:, others].float()
    order = picked.argsort(dim=1, descending=True)
    position = order[:, topk - 1:topk]
    threshold = picked.gather(1, position).squeeze(1)
    displaced = routed[:, others].gather(1, position).squeeze(1)
    return threshold, routed[:, expert] - displaced


def best_cut(thresholds, delta, bounds):
    """``const + Σ_t 1[τ(t) < knob]·delta(t)`` の、knob についての厳密な argmax。

    最大を取る knob の開区間を ``bounds`` で切って返す。``bounds`` の中の knob が
    実際に届く切り口だけを考えるので、結果は「制約なしの argmax をあとから中へ
    押し戻したもの」ではなく、許される範囲での厳密な argmax である。同点なら
    最短の接頭辞を残すので、何も得しない knob は動かない。
    """
    low, high = bounds
    if not thresholds.numel():
        return None
    order = thresholds.argsort()
    tau = thresholds[order]
    cumulative = delta[order].to(torch.float64).cumsum(0)
    # 位置 i に続く区間は tau[i] から次の閾値までであり、knob の値は**相異なる**
    # 閾値どうししか分けられない。等しい閾値の連なりの内側で終わる接頭辞には、
    # そもそも届かない。
    following = torch.cat([tau[1:], tau.new_full((1,), math.inf)])
    reachable = (following > tau) & (tau < high) & (following > low)
    positions = reachable.nonzero(as_tuple=True)[0]
    best_value = 0.0  # 空の接頭辞: このトークンを1つも選ばない
    best_index = -1
    if float(tau[0]) <= low:
        best_value = -math.inf  # ……そこへは、この knob の範囲からは届かない
    if positions.numel():
        values = cumulative[positions]
        candidate = float(values.max())
        if candidate > best_value:
            best_value = candidate
            best_index = int(positions[int((values == candidate).nonzero()[0, 0])])
    if best_index < 0:
        if best_value == -math.inf:
            return None
        return low, min(high, float(tau[0]))
    return max(low, float(tau[best_index])), min(high, float(following[best_index]))


def gain_from_cut(interval):
    """勝った区間の内側に厳密に入る正の倍率。取れなければ None。"""
    lower, upper = interval
    if not 0.0 < lower < upper or math.isinf(upper):
        return None
    return math.sqrt(lower * upper)


def bias_from_cut(interval):
    """勝った区間の内側に厳密に入る offset。取れなければ None。"""
    lower, upper = interval
    if math.isinf(lower) and math.isinf(upper):
        return None
    if math.isinf(lower):
        return upper - BIAS_BOUNDARY_MARGIN
    if math.isinf(upper):
        return lower + BIAS_BOUNDARY_MARGIN
    return (lower + upper) / 2.0


@torch.no_grad()
def calibrate_router_scores(
        expert_groups, topk, z, gate_weight, up_weight, representatives,
        router_normalized=True, chunk_size=4096, max_sweeps=10, fit_bias=True,
        gain_limit=GAIN_LIMIT):
    """代表を凍結したまま、expert ごとの gain と offset を1つずつ合わせる。

    ``expert_groups`` は先頭に shared グループを含む。shared の質量は R の分子と
    分母の両方に居て routed の選択に影響されない — 方式4 とまったく同じである。
    """
    validate_weights(gate_weight, up_weight)
    if chunk_size < 1 or max_sweeps < 1:
        raise ValueError('chunk_size と max_sweeps は正のはず')
    if not gain_limit > 1.0:
        raise ValueError(f'gain の上限 {gain_limit} は 1 の周りに余地を残さない')
    z = flatten_z(z, gate_weight.shape[1])
    n_neurons = gate_weight.shape[0]
    device = gate_weight.device

    groups = [()]
    if expert_groups and expert_groups[0]:
        groups[0] = clean_group(expert_groups[0], n_neurons, 'shared グループ')
    routed_groups = [clean_group(group, n_neurons, f'routed expert {index}')
                     for index, group in enumerate(expert_groups[1:])]
    groups.extend(routed_groups)
    n_routed = len(routed_groups)
    if n_routed < 1:
        raise ValueError('score の校正には routed expert が1個以上要る')
    if not isinstance(topk, int) or isinstance(topk, bool) or not 0 <= topk <= n_routed:
        raise ValueError(f'Top-K {topk!r} は routed expert {n_routed} 個に対して不正')
    representatives = tuple(int(index) for index in representatives)
    if len(representatives) != n_routed:
        raise ValueError(
            f'代表が {len(representatives)} 個、routed expert は {n_routed} 個')
    for expert, neuron in enumerate(representatives):
        if neuron not in routed_groups[expert]:
            raise ValueError(f'代表 {neuron} が routed expert {expert} の中にいない')

    routed, shared = expert_masses(z, groups, gate_weight, up_weight, chunk_size)
    total = float((shared + routed.sum(dim=1)).sum(dtype=torch.float64))
    if not total > 0:
        raise ValueError(f'fit の活性量が {total}')
    shared_mass = float(shared.sum(dtype=torch.float64))

    unit_gate, unit_up = unit_rows(
        gate_weight, up_weight, representatives, router_normalized)
    # ``base`` は gain=1 のときの score で、以降変わらない。gain の段の閾値は
    # これに対する絶対的な倍率なので、受理した手が自分の丸めを積み上げない。
    base = score_columns(z, unit_gate, unit_up, chunk_size)
    scores = base.clone()
    gains = [1.0] * n_routed
    biases = torch.zeros(n_routed, dtype=torch.float32, device=device)
    recovery = _recovery(
        deployed_final(scores, biases), routed, shared_mass, total, topk)
    initial_recovery = recovery
    moves = []
    sweeps_used = {'gain': 0, 'bias': 0}
    frozen = topk == 0 or topk == n_routed
    terminations = {
        'gain': 'objective-independent-of-selection' if frozen else 'converged',
        'bias': 'objective-independent-of-selection' if frozen else 'converged',
    }
    if not fit_bias and not frozen:
        terminations['bias'] = 'disabled'

    if not frozen:
        for sweep in range(max_sweeps):
            sweeps_used['gain'] = sweep + 1
            changed = False
            for expert in range(n_routed):
                # offset がまだ全部ゼロなので softmax は生の score の順序を保つ。
                # したがって gain の閾値は score 空間に居る。
                threshold, delta = _threshold_and_delta(
                    scores, routed, expert, topk)
                # score がゼロのトークンは gain が何であれ入らない。score は
                # 絶対値なので、非負の閾値を越えることが決してできない。
                usable = base[:, expert].float() > 0.0
                interval = best_cut(
                    threshold[usable] / base[usable, expert].float(), delta[usable],
                    (1.0 / gain_limit, gain_limit))
                if interval is None or interval[0] < gains[expert] < interval[1]:
                    continue  # つまみはもう勝った区間の中にいる
                gain = gain_from_cut(interval)
                if gain is None or gain == gains[expert]:
                    continue
                trial_gains = list(gains)
                trial_gains[expert] = gain
                # 行列は重みから丸ごと作り直す。決して1列ずつではない — 1行の
                # GEMM はルーターの N 行の呼び出しと同じ順序で縮約しないので、
                # 差し替えた1列は出荷されないルーターを採点することになる。
                trial = score_columns(
                    z, unit_gate, scaled_up_rows(unit_up, trial_gains), chunk_size)
                if not bool(torch.isfinite(trial).all()):
                    continue
                value = _recovery(
                    deployed_final(trial, biases), routed, shared_mass, total, topk)
                # 縮約が厳密なのは同点の割り方と行の丸めまでなので、直接測って
                # 残らなかった gain は雑音であり、その手は捨てる。
                if value <= recovery:
                    continue
                moves.append({
                    'stage': 'gain', 'sweep': sweep, 'expert': expert,
                    'from': gains[expert], 'to': gain, 'recovery': value,
                })
                scores, gains, recovery, changed = trial, trial_gains, value, True
            if not changed:
                break
        else:
            terminations['gain'] = 'max-sweeps'
    gain_recovery = recovery

    if not frozen and fit_bias:
        # ここから gain は凍結なので、softmax は1回だけ計算し、offset はその出力を
        # 直接ずらす — 配備のルーターがするとおりに。
        probabilities = scores.softmax(dim=-1, dtype=torch.float32)
        for sweep in range(max_sweeps):
            sweeps_used['bias'] = sweep + 1
            changed = False
            for expert in range(n_routed):
                threshold, delta = _threshold_and_delta(
                    probabilities + biases, routed, expert, topk)
                interval = best_cut(
                    threshold - probabilities[:, expert], delta,
                    (-math.inf, math.inf))
                current = float(biases[expert])
                if interval is None or interval[0] < current < interval[1]:
                    continue  # つまみはもう勝った区間の中にいる
                bias = bias_from_cut(interval)
                if bias is None or bias == current:
                    continue
                trial = biases.clone()
                trial[expert] = bias
                value = _recovery(
                    probabilities + trial, routed, shared_mass, total, topk)
                if value <= recovery:
                    continue
                moves.append({
                    'stage': 'bias', 'sweep': sweep, 'expert': expert,
                    'from': float(biases[expert]), 'to': bias, 'recovery': value,
                })
                biases, recovery, changed = trial, value, True
            if not changed:
                break
        else:
            terminations['bias'] = 'max-sweeps'

    return CalibrationSelection(
        representatives=representatives,
        gains=tuple(gains),
        biases=tuple(float(value) for value in biases),
        recovery=recovery,
        initial_recovery=initial_recovery,
        gain_recovery=gain_recovery,
        moves=tuple(moves),
        sweeps_used=sweeps_used,
        terminations=terminations,
        gain_limit=gain_limit,
        faithfulness=order_faithfulness(scores, topk),
        n_tokens=z.shape[0],
    )


class ScoreCalibrationMethod:
    name = 'score_calibration'
    training_free = True
    requires_source_weights = True
    requires_fit_z = True
    requires_initial_sets = True
    diagnostic_only = False
    # 代表を1つも動かさないので、出発点は「どれかの方式が選んだ集合」ではなく
    # **凍結する相手そのもの**である。組み立て役はこの名前を見て1本だけ渡す。
    frozen_source = 'oracle_recovery'

    def __init__(self, chunk_size=4096, max_sweeps=10, fit_bias=True,
                 gain_limit=GAIN_LIMIT):
        self.chunk_size = chunk_size
        self.max_sweeps = max_sweeps
        self.fit_bias = fit_bias
        self.gain_limit = gain_limit

    def build(self, context, baseline):
        sets = context.initial_representative_sets
        if len(sets) != 1:
            raise ValueError(
                f'score の校正が凍結するのは代表集合1本。{len(sets)} 本受け取った')
        selection = calibrate_router_scores(
            context.expert_groups,
            context.topk,
            context.fit_z,
            context.gate_weight,
            context.up_weight,
            sets[0],
            router_normalized=context.router_normalized,
            chunk_size=self.chunk_size,
            max_sweeps=self.max_sweeps,
            fit_bias=self.fit_bias,
            gain_limit=self.gain_limit,
        )
        router = copy.deepcopy(baseline)
        gate_rows, up_rows = unit_rows(
            context.gate_weight, context.up_weight,
            selection.representatives, context.router_normalized)
        router.gate.weight.data.copy_(gate_rows)
        router.classifier.weight.data.copy_(scaled_up_rows(up_rows, selection.gains))
        router.extra_bias.copy_(
            torch.tensor(selection.biases, dtype=router.extra_bias.dtype,
                         device=router.extra_bias.device))
        router.representative_indices = selection.representatives
        router.selection = {
            'rule': 'coordinate-ascent-on-per-expert-score-gain-and-offset',
            **selection.metadata(),
        }
        return router
