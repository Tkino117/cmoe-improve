"""Llama 系（LlamaForCausalLM）のアダプタ。

CMoE-ref の ``run_cmoe.get_llama`` / ``catch_first_inputs`` /
``xsearch.capture.LayerCapture`` の attention 部分を、モデル差異の境界の内側に
まとめたもの。

transformers を 4.47.1 に固定しているのは、層を ``position_embeddings`` 抜きで
呼ぶため。4.48 で ``LlamaAttention.rotary_emb`` のフォールバックが消えており、
過去の測定と同じ呼び方ができなくなる。
"""

import torch
import torch.nn as nn

from cmoe.adapters.base import DenseFFN, LayerInputs
from cmoe.moe.modules import MoE

DEFAULT_SEQLEN = 2048


def _slice_batch(value, start, stop, batch_size):
    """バッチ次元を持つものだけを切る。mask/position_ids は共有のことがある。"""
    if isinstance(value, torch.Tensor) and value.dim() and value.shape[0] == batch_size:
        return value[start:stop]
    return value


class LlamaAdapter:

    def __init__(self, model, seqlen=DEFAULT_SEQLEN, device=None):
        self.model = model
        self.seqlen = seqlen
        self.device = torch.device(device or 'cuda:0')
        self.model.seqlen = seqlen

    @classmethod
    def load(cls, name_or_path, seqlen=DEFAULT_SEQLEN, device=None,
             dtype=torch.bfloat16):
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map='auto')
        model.eval()
        model.config.use_cache = False
        return cls(model, seqlen=seqlen, device=device)

    # -- 構造 -------------------------------------------------------------

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def dtype(self):
        return next(iter(self.model.parameters())).dtype

    def to_device(self):
        self.model.to(self.device)
        return self

    def is_converted(self, index):
        return isinstance(self.layers[index].mlp, MoE)

    def dense_ffn(self, index):
        mlp = self.layers[index].mlp
        if isinstance(mlp, MoE):
            raise ValueError(f'層 {index} はすでに変換済み。層の変換は一度だけ')
        return DenseFFN(
            hidden_size=mlp.hidden_size,
            intermediate_size=mlp.intermediate_size,
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            act_fn=mlp.act_fn,
        )

    def replace_ffn(self, index, module):
        self.layers[index].mlp = module

    # -- 前向き -----------------------------------------------------------

    @torch.no_grad()
    def capture_layer_inputs(self, input_ids):
        """層0 の入力を捕まえる。捕まえた時点で例外を投げて前向きを止める。"""
        layers = self.layers
        cache = {}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                cache['hidden'] = inp.detach().cpu().clone()
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers[0] = Catcher(layers[0])
        try:
            self.model(input_ids.to(self.device))
        except ValueError:
            pass
        finally:
            layers[0] = layers[0].module
        if 'hidden' not in cache:
            raise ValueError('層0 の入力を捕まえられなかった')
        return LayerInputs(cache['hidden'], cache['attention_mask'],
                           cache['position_ids'])

    @torch.no_grad()
    def forward_attention(self, index, hidden, attention_mask, position_ids,
                          batch_chunk=None):
        """attention 側だけを進め、(z, residual) を返す。

        戻り値は入力と同じデバイス（CPU の隠れ状態を渡せば CPU で返る）。
        ``batch_chunk`` を指定するとバッチを分割して進めるので、64 系列でも
        中間テンソルが GPU に載る。
        """
        layer = self.layers[index]
        batch_size = hidden.shape[0]
        step = batch_chunk or batch_size
        home = hidden.device
        zs, residuals = [], []
        for start in range(0, batch_size, step):
            stop = min(start + step, batch_size)
            residual = hidden[start:stop].to(self.device)
            states = layer.input_layernorm(residual)
            states = layer.self_attn(
                hidden_states=states,
                attention_mask=_slice_batch(attention_mask, start, stop, batch_size),
                position_ids=_slice_batch(position_ids, start, stop, batch_size),
            )[0]
            residual = residual + states
            z = layer.post_attention_layernorm(residual)
            zs.append(z.to(home))
            residuals.append(residual.to(home))
        return torch.cat(zs, dim=0), torch.cat(residuals, dim=0)

    @torch.no_grad()
    def forward_layer(self, index, hidden, attention_mask, position_ids):
        return self.layers[index](
            hidden, attention_mask=attention_mask, position_ids=position_ids)[0]

    @torch.no_grad()
    def head(self, hidden):
        norm = self.model.model.norm
        if norm is not None:
            hidden = norm(hidden)
        return self.model.lm_head(hidden)
