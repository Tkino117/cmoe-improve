"""アダプタ名の解決。CLI に分岐を持たせないための唯一の場所。"""

from cmoe.adapters.auto import AutoAdapter
from cmoe.adapters.llama import LlamaAdapter

ADAPTERS = {
    # 既存の測定を再現する経路。LlamaForCausalLM を名指しで読む
    'llama': LlamaAdapter,
    # 同じ層構造の他モデル（Mistral・Qwen2 など）を AutoModelForCausalLM で読む
    'auto': AutoAdapter,
}

# モデル名からアダプタを推測するときの手掛かり。当たらなければ --adapter で明示する。
_HINTS = (
    ('llama', 'llama'),
    ('vicuna', 'llama'),
    ('mistral', 'auto'),
    ('qwen', 'auto'),
)


def create_adapter(name, model, **kwargs):
    try:
        adapter = ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f'未知のアダプタ {name!r}。{sorted(ADAPTERS)} から選ぶ') from None
    return adapter.load(model, **kwargs)


def guess_adapter(model):
    lowered = model.lower()
    for hint, name in _HINTS:
        if hint in lowered:
            return name
    raise ValueError(
        f'{model!r} のアダプタを推測できない。--adapter で明示する '
        f'({sorted(ADAPTERS)})')
