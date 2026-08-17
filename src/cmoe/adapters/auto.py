"""decoder 型 causal LM 全般のアダプタ。

``LlamaAdapter`` のうち Llama 固有なのは、読み込みに ``LlamaForCausalLM`` を
名指ししている点だけである。層の並び (``model.model.layers``)、層内の名前
(``input_layernorm`` / ``post_attention_layernorm`` / ``mlp.{gate,up,down}_proj``)、
最終段 (``model.model.norm`` + ``lm_head``) は Llama・Mistral・Qwen2 で共通なので、
読み込みを ``AutoModelForCausalLM`` に替えるだけで通る。

Llama 用は別に残してある。既存の測定を再現する経路を、他モデル対応の変更で
動かしたくないためである。この形に当てはまらないモデル（FFN が gate/up/down で
ない、層の置き場所が違う）は、ここを継承せず別のアダプタを書く。
"""

import torch

from cmoe.adapters.llama import DEFAULT_SEQLEN, LlamaAdapter


class AutoAdapter(LlamaAdapter):
    """AutoModelForCausalLM で読める、Llama と同じ層構造のモデル。"""

    @classmethod
    def load(cls, name_or_path, seqlen=DEFAULT_SEQLEN, device=None,
             dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map='auto')
        model.eval()
        model.config.use_cache = False
        adapter = cls(model, seqlen=seqlen, device=device)
        adapter.check_shape()
        return adapter

    def check_shape(self):
        """想定した構造かどうかを、読み込み直後に確かめる。

        当てはまらないモデルを黙って変換すると、失敗するのは数十分後の
        評価の途中になる。
        """
        layer = self.layers[0]
        for name in ('input_layernorm', 'post_attention_layernorm', 'mlp',
                     'self_attn'):
            if not hasattr(layer, name):
                raise ValueError(
                    f'{type(layer).__name__} に {name} が無い。'
                    'このモデルには専用のアダプタが要る')
        for name in ('gate_proj', 'up_proj', 'down_proj', 'act_fn'):
            if not hasattr(layer.mlp, name):
                raise ValueError(
                    f'{type(layer.mlp).__name__} に {name} が無い。'
                    'CMoE は gate/up/down の三つ組を前提にしている')
        return self
