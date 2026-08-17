"""ルーター方式名の解決。CLI に方式の分岐を持たせないための唯一の場所。"""

from cmoe.router.methods.cmoe import CMoEMethod

METHODS = {
    'cmoe': CMoEMethod,
}


def create_method(name):
    try:
        method = METHODS[name]
    except KeyError:
        raise ValueError(
            f'未知のルーター方式 {name!r}。{sorted(METHODS)} から選ぶ') from None
    return method()
