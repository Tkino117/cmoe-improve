"""採点オラクル名の解決。CLI にオラクルの分岐を持たせないための唯一の場所。

安い順に並んでいる。層ローカルの2つは後続の層を1つも走らせず、``suffix_kl`` は
子1つにつき残りの層すべてを走らせる。「計算量を落とす探索」の実験は、この段差の
上で探索を差し替えて比べることになる。
"""

from cmoe.alloc.oracles.local_error import LocalErrorOracle
from cmoe.alloc.oracles.mass import MassOracle
from cmoe.alloc.oracles.suffix_kl import SuffixKLOracle

ORACLES = {
    # 層ローカル・最安: 取りこぼした |h| 質量の割合
    'mass': lambda walk: MassOracle(walk, squared=False),
    # 同じく層ローカル。h^2 重み = 出力誤差の対角近似
    'mass_squared': lambda walk: MassOracle(walk, squared=True),
    # 層ローカル: その層の相対二乗出力誤差 L(x)
    'local_error': lambda walk: LocalErrorOracle(walk),
    # 高価: 残りを dense のまま走らせた出力分布の KL。ビーム配分の出典
    'suffix_kl': lambda walk: SuffixKLOracle(walk),
}


def create_oracle(name, walk):
    check_oracle(name)
    return ORACLES[name](walk)


def check_oracle(name):
    """名前だけを先に検査する。オラクルの生成は walk を要求するので分けてある。

    モデルを読み込む前に呼ぶためにある。7B を読み終えてから引数の綴り違いで
    落ちると、十数分が引数エラーのために消える。
    """
    if name not in ORACLES:
        raise ValueError(f'未知の採点オラクル {name!r}。{sorted(ORACLES)} から選ぶ')
