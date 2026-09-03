"""ルーター方式名の解決。CLI に方式の分岐を持たせないための唯一の場所。"""

from cmoe.router.methods.cmoe import CMoEMethod
from cmoe.router.methods.dynamic_topk import (DynamicSpectralMethod,
                                             DynamicTopKMethod)
from cmoe.router.methods.expert_mean import (CenteredExpertMeanMethod,
                                             ExpertMeanMethod)
from cmoe.router.methods.frequency_centroid import FrequencyCentroidMethod
from cmoe.router.methods.oracle_abs import OracleAbsMethod
from cmoe.router.methods.oracle_correlation import OracleCorrelationMethod
from cmoe.router.methods.oracle_recovery import OracleRecoveryMethod
from cmoe.router.methods.score_calibration import ScoreCalibrationMethod
from cmoe.router.methods.spectral_mass import (PlainSpectralMassMethod,
                                              SpectralMassMethod,
                                              WeightOnlySpectralMassMethod)

METHODS = {
    # 方式1: 現行 CMoE。すべての実験の対照
    'cmoe': CMoEMethod,
    # 方式2: 活性頻度で重み付けした重心
    'freq_centroid': FrequencyCentroidMethod,
    # 方式3: expert 質量との Pearson 相関が最大の代表
    'oracle_correlation': OracleCorrelationMethod,
    # 方式4: 回収率の共同最適化（本命）
    'oracle_recovery': OracleRecoveryMethod,
    # 方式5: 代表を凍結し、expert ごとの score の gain と offset を合わせる
    'score_calibration': ScoreCalibrationMethod,
    # 方式6: expert の行の平均
    'expert_mean': ExpertMeanMethod,
    # 方式7: 重み行列の低ランク近似から expert の活性質量を見積もる。
    # 代表ニューロン1本の族を出る最初の方式
    'spectral_mass': SpectralMassMethod,
    # 同じ方式で白色化を切ったもの。基底が校正データに依らない
    'spectral_plain': PlainSpectralMassMethod,
    # さらに silu の行の重みも切ったもの。校正データを読まない
    'spectral_weight': WeightOnlySpectralMassMethod,
    # 方式8: 固定 Top-K をやめ、平均だけを守ってトークン間で配り直す
    'dynamic_cmoe': DynamicTopKMethod,
    'dynamic_spectral': DynamicSpectralMethod,
    'expert_mean_centered': CenteredExpertMeanMethod,
    # 診断専用: 真の |h| を読むオラクル。配備できない
    'oracle_abs': OracleAbsMethod,
}

# 方式4 は先行方式の代表集合から座標上昇を始めるので、この順に構築する。
# 途中の方式を飛ばすと出発点が減り、別の答えになる。
RECOVERY_CHAIN = ('cmoe', 'freq_centroid', 'oracle_correlation', 'oracle_recovery')


def create_method(name):
    """方式名を実体にする。``name:値`` で方式の主パラメータを添えられる。

    ``spectral_mass:64`` のように書くと、その方式が ``variant_attribute`` で
    名乗った属性へ値が入る。同じ方式のパラメータ違いを**1回の変換の中で**
    並べるためにある — ルーターだけを差し替えて比べるのが report/10・18 の
    やり方で、rank を CLI の引数にすると run を分けることになり、比較が
    変換をまたいでしまう。
    """
    base, separator, variant = name.partition(':')
    try:
        method = METHODS[base]
    except KeyError:
        raise ValueError(
            f'未知のルーター方式 {base!r}。{sorted(METHODS)} から選ぶ') from None
    instance = method()
    if separator:
        attribute = getattr(method, 'variant_attribute', None)
        if attribute is None:
            raise ValueError(f'方式 {base!r} はパラメータ付きの名前を取らない')
        try:
            value = int(variant)
        except ValueError:
            raise ValueError(
                f'{name!r} のパラメータ {variant!r} が整数でない') from None
        setattr(instance, attribute, value)
        # 記録も比較も名前で引くので、変種は変種の名前を名乗る
        instance.name = name
    return instance


def resolve_chain(names):
    """要求された方式を、依存を満たす構築順に並べ直す。

    ``oracle_recovery`` は初期代表集合を要求するので、先行3方式を前に挿す。
    ``frozen_source`` を持つ方式（方式5）は凍結する相手そのものを要求するので、
    その方式を前に挿す。それ以外の方式は互いに独立なので、要求された順のまま
    後ろに置く。
    """
    requested = list(dict.fromkeys(names))
    # 凍結型の方式は、凍結する相手が先に居ないと作れない
    needed = list(requested)
    for name in requested:
        frozen = getattr(METHODS.get(name), 'frozen_source', None)
        if frozen is not None and frozen not in needed:
            needed.insert(needed.index(name), frozen)
    order = []
    if 'oracle_recovery' in needed:
        order.extend(RECOVERY_CHAIN)
    for name in needed:
        if name not in order:
            order.append(name)
    if 'cmoe' not in order:
        # 対照であり、carve の伝播に使うルーターでもあるので必ず先頭に置く
        order.insert(0, 'cmoe')
    elif order[0] != 'cmoe':
        order.remove('cmoe')
        order.insert(0, 'cmoe')
    return order
