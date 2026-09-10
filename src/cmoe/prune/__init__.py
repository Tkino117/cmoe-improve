"""[対照] 静的な構造化プルーニング。MoE 変換を通らない比較手法。

``base`` が計画とパラメータ会計、``flap`` / ``llm_pruner`` が計画の作り方、
``registry`` が名前の解決。**どれも提案手法ではない。**
"""

from cmoe.prune.base import (LayerPlan, PrunePlan, accounting, apply_plan,
                             model_shape, moe_activated_sparsity)

__all__ = ['LayerPlan', 'PrunePlan', 'accounting', 'apply_plan', 'model_shape',
           'moe_activated_sparsity']
