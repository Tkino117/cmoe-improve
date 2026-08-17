"""ルーター方式の実装。"""

from cmoe.router.methods.cmoe import CMoEMethod
from cmoe.router.methods.expert_mean import (CenteredExpertMeanMethod,
                                             ExpertMeanMethod)
from cmoe.router.methods.frequency_centroid import FrequencyCentroidMethod
from cmoe.router.methods.oracle_abs import OracleAbsMethod
from cmoe.router.methods.oracle_correlation import OracleCorrelationMethod
from cmoe.router.methods.oracle_recovery import OracleRecoveryMethod

__all__ = ['CMoEMethod', 'FrequencyCentroidMethod', 'OracleCorrelationMethod',
           'OracleRecoveryMethod', 'ExpertMeanMethod',
           'CenteredExpertMeanMethod', 'OracleAbsMethod']
