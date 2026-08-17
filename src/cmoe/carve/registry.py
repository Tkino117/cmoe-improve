"""分割方式名の解決。"""

from cmoe.carve.cmoe import CMoECarver

CARVERS = {
    'cmoe': CMoECarver,
}


def create_carver(name, n_experts):
    try:
        carver = CARVERS[name]
    except KeyError:
        raise ValueError(
            f'未知の分割方式 {name!r}。{sorted(CARVERS)} から選ぶ') from None
    return carver(n_experts)
