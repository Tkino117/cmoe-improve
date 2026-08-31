"""接頭辞を、選択問題の**採点そのもの**で採点する。

``suffix_kl`` が測るのは親モデルとの近さである。dense がスコア 0 で、そこから
離れた量が点になる。それは「壊さない」ことの指標であって、「当たる」ことの指標
ではない — report/11 と report/14 が繰り返し出したのは、その2つの順位が一致
しないという結果だった（report/14 段2 では、beam3 が KL で一様配分に勝ちながら
``acc`` で 0.6113 対 0.6170 と負けている）。

ここが測るのは、lm-eval が正誤を決めるときに見ているものそのものである。選択肢
K 本それぞれの対数尤度を取り、正解と不正解の差 — マージン — を作る:

    M = -log p(正解) - softmin_β( -log p(不正解 1), ..., -log p(不正解 K-1) )
      = -ll(正解) + (1/β) logsumexp( β · ll(不正解) )

**小さいほど良い。** 正解の対数尤度が高く、いちばん惜しい不正解の対数尤度が
低いほど、M は小さくなる。オラクルの契約（小さいほど良い）にそのまま乗る。

**softmin が入る理由。** 素の min（β → ∞）は最も惜しい不正解1本しか見ないので、
2位以下がどれだけ迫っていても点が動かない。β を有限にすると、近い不正解ほど
大きく効く滑らかな min になる。

**β の両端に、既にある指標が立っている。**

* β → ∞ で M は ``bench_stats.margin`` の符号反転そのものになる（正解 −
  最良の不正解）。その符号が ``acc`` である
* β = 1 では M = log( Σ_不正解 p / p_正解 ) になり、さらに ``softplus(M)`` を
  取ると ``bench_stats.gold_nll``（選択肢上で softmax した正解確率の −log）と
  **厳密に一致する**

つまり ``--margin-beta`` と ``--margin-loss`` が動かしているのは、評価側に既に
ある2つの指標を両端に持つ族の中の一点である。校正の目的関数と評価指標が同じ
族から出ることは、この線の主張そのものでもある。

**損失の形（``loss``）。**

* ``raw``      M の平均。書いたとおりのマージン。下に有界でない
* ``softplus`` log(1 + exp(M)) の平均。0 で下に有界で、既に当たっている問題の
  寄与が飽和する。β = 1 では ``gold_nll`` と一致する

**タスクをまたぐ束ね方はマクロ平均である。** ベンチの集計（``bench_stats.summarize``）
がタスクを等しく重み付けるので、目的関数も同じ束ね方にする。校正セットは
タスクごとに採点位置を等しく配っているが、問題数は選択肢の長さのぶん揃わない。

**dense は目標ではない。** ここが ``suffix_kl`` との一番大きな違いである。KL は
dense がスコア 0 の下限を持ち、非決定性の床と比べて「区別できる差か」を言えた。
マージンにその下限は無く、**dense より良いスコアの配分が原理的に存在する**。
それが狙いだが、同時に校正の問題への過適合と区別が付かないということでもある
ので、``reference_score()`` が dense の値を参照点として残す。

**接頭辞のスコアが何であるか**は ``suffix_kl`` と同じである。層 ℓ を測るとき
ℓ+1 以降は dense のままなので、スコアは「残りが dense のままならこの接頭辞は
いくら当たるか」であり、最終層でだけ実在するモデルの値になる。
"""

import torch

from cmoe.alloc.base import ScoreResult
from cmoe.alloc.oracles.base import PrefixOracleBase

# log_softmax 1回あたりの採点位置数。fp32 の [位置, 語彙] がここで一番大きい
# 確保になるので、塊で進める（``suffix_kl`` と同じ理由・同じ既定）。
TOKEN_CHUNK = 1024

# 目的関数の形。名前は CLI の ``--margin-loss`` がそのまま取る
LOSSES = ('raw', 'softplus')


@torch.no_grad()
def sequence_loglikelihoods(logits, scoring, token_chunk=TOKEN_CHUNK):
    """採点位置の読み出しから、系列ごとの対数尤度 [n_rows]。

    lm-eval が選択肢1本に付ける点そのもの — 続きのトークンの対数確率の**和**
    である（長さで割らない。``acc`` と ``margin`` がその生の和を比べている）。

    ``logits`` は [採点位置, 語彙] で、並びは ``scoring.targets`` と同じ行優先
    である。塊ごとに log_softmax を取り、正解トークンの1列だけを拾って、系列
    ごとに足し込む。

    足し込みは fp64 で行う。1系列あたり数十項なので桁落ちはしないが、系列の
    対数尤度の差が候補の順序を決める量なので、和の取り方を揺らさない。
    """
    if logits.shape[0] != scoring.n_positions:
        raise ValueError(
            f'読み出しは {logits.shape[0]} 位置、印は {scoring.n_positions} 位置 '
            '— 対応していない')
    totals = torch.zeros(scoring.n_rows, dtype=torch.float64,
                         device=logits.device)
    targets = scoring.targets.to(logits.device)
    rows = scoring.row_index.to(logits.device)
    for start in range(0, scoring.n_positions, token_chunk):
        stop = min(start + token_chunk, scoring.n_positions)
        log_p = logits[start:stop].to(torch.float32).log_softmax(dim=-1)
        picked = log_p.gather(1, targets[start:stop].unsqueeze(1)).squeeze(1)
        totals.index_add_(0, rows[start:stop], picked.to(torch.float64))
        del log_p, picked
    return totals


@torch.no_grad()
def margins(row_loglikelihoods, scoring, beta):
    """問題ごとのマージン M [問題]。小さいほど良い。

        M = -ll(正解) + (1/β) logsumexp( β · ll(不正解) )

    有効でない升（K が問題ごとに違うので右が埋まっている）と正解肢の升を
    −inf にしてから logsumexp を取る。β を掛けても −inf は −inf のままで、
    不正解が必ず1本以上あることは ``ChoiceGroups`` が作った時点で保証されて
    いる。
    """
    if not beta > 0:
        raise ValueError(f'β は正の数（{beta}）')
    device = row_loglikelihoods.device
    rows = scoring.choice_rows.to(device)
    mask = scoring.choice_mask.to(device)
    gold = scoring.gold_column.to(device)

    values = row_loglikelihoods[rows]
    gold_ll = values.gather(1, gold.unsqueeze(1)).squeeze(1)
    wrong = mask.clone()
    wrong[torch.arange(wrong.shape[0], device=device), gold] = False
    masked = values.masked_fill(~wrong, float('-inf'))
    soft_best = torch.logsumexp(beta * masked, dim=1) / beta
    return soft_best - gold_ll


def apply_loss(values, loss):
    """マージンを目的関数の項にする。``raw`` はそのまま、``softplus`` は潰す。"""
    if loss == 'raw':
        return values
    if loss == 'softplus':
        return torch.nn.functional.softplus(values)
    raise ValueError(f'未知の損失 {loss!r}。{list(LOSSES)} から選ぶ')


def macro_mean(values, scoring):
    """タスクごとに平均してから、タスクを等しく平均する。

    問題数がタスクで揃わないので、まとめて平均すると選択肢の短いタスクの話に
    なる。ベンチの集計と同じ束ね方にしてある。
    """
    index = scoring.task_index.to(values.device)
    n_tasks = len(scoring.task_names)
    totals = torch.zeros(n_tasks, dtype=values.dtype, device=values.device)
    counts = torch.zeros(n_tasks, dtype=values.dtype, device=values.device)
    totals.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones_like(values))
    per_task = totals / counts
    return float(per_task.mean()), per_task


class MarginOracle(PrefixOracleBase):
    """残りの層を dense のまま走らせて、選択問題のマージンを測る。

    コストは ``suffix_kl`` と同じ単位・同じ量である（子1つにつき、自分の層と
    後続の層すべての forward）。違うのは読み出しから何を作るかだけなので、
    「同じ予算で目的関数を替えると何が取れるか」がそのまま比べられる。

    読み出しは採点位置だけを持つ（``keep``）。語彙の幅を2枚並べる KL と違い、
    塊ごとに1列を拾って捨てるので、確保は ``suffix_kl`` より小さい。
    """

    name = 'margin'
    cost_unit = 'layer_forwards'

    def __init__(self, walk, beta=None, loss=None, token_chunk=TOKEN_CHUNK,
                 batch_chunk=None):
        super().__init__(walk)
        if walk.choices is None:
            raise ValueError(
                'margin は選択肢ごとの系列を持つ校正セットが要る'
                '（--calib benchchoice）。1問1系列の benchqa は正解肢しか'
                '持たないので、比べる相手が無い')
        self.scoring = walk.choices
        # β の既定は 1。そこで softplus を掛けると gold_nll と厳密に一致する
        self.beta = 1.0 if beta is None else float(beta)
        self.loss = 'raw' if loss is None else loss
        if self.loss not in LOSSES:
            raise ValueError(f'未知の損失 {self.loss!r}。{list(LOSSES)} から選ぶ')
        if not self.beta > 0:
            raise ValueError(f'--margin-beta は正の数（{self.beta}）')
        self.token_chunk = token_chunk
        self.batch_chunk = batch_chunk if batch_chunk is not None else walk.batch_chunk
        self._dense = None

    @torch.no_grad()
    def score_logits(self, logits):
        """読み出し [採点位置, 語彙] から (スコア, 内訳)。"""
        row_ll = sequence_loglikelihoods(logits, self.scoring, self.token_chunk)
        values = margins(row_ll, self.scoring, self.beta)
        score, per_task = macro_mean(apply_loss(values, self.loss), self.scoring)
        # 正答率も一緒に出す。マージンの符号がそれなので、追加の forward は
        # 要らない。探索が目的関数を下げながら acc を落としていないかを、
        # 走っている最中に見られる
        accuracy, _ = macro_mean((values < 0).to(values.dtype), self.scoring)
        details = {
            'margin': score,
            'beta': self.beta,
            'loss': self.loss,
            'calibration_acc': accuracy,
            'per_task': {name: float(value) for name, value
                         in zip(self.scoring.task_names, per_task)},
            'n_questions': self.scoring.n_questions,
        }
        return score, details

    @torch.no_grad()
    def run_suffix(self, start_layer, hidden):
        return self.walk.adapter.forward_suffix(
            start_layer, hidden, self.walk.inputs,
            batch_chunk=self.batch_chunk, keep=self.scoring.keep)

    @torch.no_grad()
    def reference_score(self):
        """dense モデルのスコア。**目標ではなく参照点である。**

        KL と違ってこれは下限ではない — 配分探索がここを下回ることは起こりうる
        し、それがこの目的関数に替えた狙いでもある。下回ったときにそれが本物か
        校正への過適合かは、校正に入っていない問題で測って初めて分かる。
        """
        if self._dense is None:
            state = self.walk.root()
            logits = self.run_suffix(0, state.hidden)
            self._dense = self.score_logits(logits)
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self._dense[0]

    @torch.no_grad()
    def reference_details(self):
        """dense の内訳（タスクごとのマージンと、校正上の正答率）。

        参照点はスコア1つでは読めない。探索が目的関数を下げながら正答率を
        落としていないかは、dense の正答率と並べて初めて言えるので、記録に
        残すのはこちらである。
        """
        self.reference_score()
        return self._dense[1]

    @torch.no_grad()
    def nondeterminism_floor(self):
        """同じ dense の読み出しを2回取ったときの、スコアの差の絶対値。

        重みも forward も同じなので、2つを隔てているのはこの機械のカーネルの
        非決定性だけである。これより小さい差で2つの配分を区別することはできない。
        """
        reference = self.reference_score()
        state = self.walk.root()
        logits = self.run_suffix(0, state.hidden)
        repeated, _ = self.score_logits(logits)
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return abs(repeated - reference)

    @torch.no_grad()
    def measure(self, profile, carved, state, child):
        logits = self.run_suffix(profile.layer + 1, child.hidden)
        score, details = self.score_logits(logits)
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 自分の層と、そのあとに走らせた層。``suffix_kl`` と同じ数え方なので、
        # 2つのオラクルのコストは同じ物差しの上に乗る
        cost = float(self.walk.n_layers - profile.layer)
        details.update({'layer': profile.layer, 'x': carved.n_shared,
                        'topk': carved.topk,
                        'dense_suffix_layers':
                            self.walk.n_layers - profile.layer - 1})
        return ScoreResult(score=score, cost=cost, cost_unit=self.cost_unit,
                           details=details)
