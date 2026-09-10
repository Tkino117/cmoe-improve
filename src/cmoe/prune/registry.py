"""プルーニング手法名の解決。CLI に分岐を持たせないための唯一の場所。"""

from cmoe.prune import flap, llm_pruner

METHODS = {
    'flap': flap,
    'llm_pruner': llm_pruner,
}


def create_plan(name, adapter, calibration, sparsity, scope, log=None, **kwargs):
    try:
        module = METHODS[name]
    except KeyError:
        raise ValueError(
            f'未知のプルーニング手法 {name!r}。{sorted(METHODS)} から選ぶ') from None
    if name == 'llm_pruner':
        kwargs.setdefault('log', log)
    else:
        kwargs.setdefault('progress', None)
    return module.build_plan(adapter, calibration, sparsity, scope=scope, **kwargs)


def calibration_defaults(name):
    """原典の既定の校正予算（本数, 系列長）。ソースは呼ぶ側が決める。"""
    module = METHODS[name]
    return module.DEFAULT_SAMPLES, module.DEFAULT_SEQLEN
