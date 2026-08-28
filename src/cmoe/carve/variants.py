"""現行 CMoE の分割規則を1箇所ずつ動かした派生。

対照（``cmoe`` ）との差を1つに保つのが目的である。3つとも
``carve/cmoe.py`` の ``CMoECarver`` を継承し、差し替えるのは1メソッド
（あるいは1定数）だけにしてある。

* ``cmoe_iter5``   … クラスタ割り当ての反復を 1 → 5 にする。移送元が1回で
                     止めているのは実装の都合であって、結論ではない
* ``cmoe_weighted``… クラスタリングの特徴を 0/1 から ``|h|`` にする。取りこぼす
                     量は大きさで決まるのに、現行は大きさを見ていない
* ``cmoe_mass``    … shared に抜く順位を活性頻度から質量 ``Σ_t |H|`` にする。
                     稀にしか上位に来ないが来たときに大きいニューロンは、
                     頻度では拾われない

いずれも回収率（``mass`` / ``local_error``）を下げにいく変更で、それが PPL や
選択問題の正答率に伝わるかどうかは別問題である。report/10 は隣の軸（ルーター）
で「回収率を上げるほどベンチが悪くなる」を観測している。
"""

from cmoe.carve.cmoe import CMoECarver
from cmoe.carve.profile import marker_weights, neuron_mass


class IteratedCMoECarver(CMoECarver):
    """反復回数だけを変えたもの。特徴も shared の選び方も現行のまま。"""

    name = 'cmoe_iter5'
    max_iters = 5


class WeightedMarkerCarver(CMoECarver):
    """クラスタリングの特徴を ``|h|`` に替えたもの。

    印の位置は現行と同じで、値が 1.0 から ``|h|`` に変わる。L1 距離は
    「どちらか一方だけが上位に入ったトークン数」から「上位に入った量の差」に
    なる。shared の選び方（活性頻度）は変えない。

    プロファイルは markers を作るときに h を捨てているので、ここで作り直す。
    層あたり FFN の前向き1回ぶんの追加である。
    """

    name = 'cmoe_weighted'

    def __init__(self, n_experts, profiling_norm=True, batch_chunk=None):
        super().__init__(n_experts)
        self.profiling_norm = profiling_norm
        self.batch_chunk = batch_chunk

    def _features(self, dense, rates, markers, z):
        if z is None:
            raise ValueError(
                f'{self.name} は z（層の FFN 入力）が要る。'
                '渡していない呼び出し経路がある')
        return marker_weights(dense, z, k_act=self.k_act,
                              normalize=self.profiling_norm,
                              batch_chunk=self.batch_chunk,
                              device=dense.gate_proj.weight.device)


class MassSharedCarver(CMoECarver):
    """shared を質量で選ぶもの。クラスタリングは現行のまま。

    現行は「上位10に入った**頻度**」で shared を切る。1票ずつなので、稀な
    トークンでだけ大きく効くニューロンは、多くのトークンでそこそこ効く
    ニューロンに必ず負ける。ここは ``Σ_t |H_ti|`` で切る。

    クラスタ中心の種は現行どおり活性頻度で選ぶ。変えると「shared の選び方を
    変えた」のか「初期値を変えた」のか分からなくなる。
    """

    name = 'cmoe_mass'

    def __init__(self, n_experts, batch_chunk=None):
        super().__init__(n_experts)
        self.batch_chunk = batch_chunk

    def _shared_scores(self, dense, rates, markers, z):
        if z is None:
            raise ValueError(
                f'{self.name} は z（層の FFN 入力）が要る。'
                '渡していない呼び出し経路がある')
        return neuron_mass(dense, z, batch_chunk=self.batch_chunk,
                           device=dense.gate_proj.weight.device)
