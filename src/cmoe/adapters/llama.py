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

from cmoe.adapters.base import DenseFFN, LayerInputs, slice_batch
from cmoe.moe.modules import MoE

DEFAULT_SEQLEN = 2048


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
                attention_mask=slice_batch(attention_mask, start, stop, batch_size),
                position_ids=slice_batch(position_ids, start, stop, batch_size),
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
    def forward_suffix(self, start_layer, hidden, inputs, batch_chunk=None,
                       keep=None):
        """層 ``start_layer`` 以降を走らせて logits にする。

        start_layer が層数と等しければ空の suffix で、最終 norm と lm_head だけが
        走る。最終層まで変換した接頭辞にはこれが要るので、呼ぶ側の分岐にしない。

        hidden は読むだけで書き換えない（層はどれも新しいテンソルを返す）ので、
        呼ぶ側の状態は次の候補にそのまま使える。

        ``keep`` を渡したときは、その印の付いた位置だけを [位置, 語彙] で返す。
        選ぶのは塊ごとに、head を当てた直後である — 捨てる位置の logits が全長で
        並ぶ瞬間を作らないためで、これが確保の上限を決める。**走らせる位置は
        減らない**（因果 attention では、後ろの位置が前の位置を読む）。
        """
        if not 0 <= start_layer <= self.n_layers:
            raise ValueError(
                f'start_layer={start_layer} は 0..{self.n_layers} の外')
        if hidden.dim() != 3:
            raise ValueError(f'[bsz, seq, hidden] のはず（{tuple(hidden.shape)}）')
        if keep is not None and tuple(keep.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                f'印は {tuple(keep.shape)}、隠れ状態は {tuple(hidden.shape[:2])} — '
                'この接頭辞の入力と対応していない')

        batch_size = hidden.shape[0]
        step = batch_chunk or batch_size
        chunks = []
        for start in range(0, batch_size, step):
            stop = min(start + step, batch_size)
            states = hidden[start:stop].to(self.device)
            mask = slice_batch(inputs.attention_mask, start, stop, batch_size)
            positions = slice_batch(inputs.position_ids, start, stop, batch_size)
            for index in range(start_layer, self.n_layers):
                states = self.forward_layer(index, states, mask, positions)
            logits = self.head(states)
            if keep is not None:
                logits = logits[keep[start:stop].to(logits.device)]
            chunks.append(logits)
            del states, logits
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    @torch.no_grad()
    def head(self, hidden):
        norm = self.model.model.norm
        if norm is not None:
            hidden = norm(hidden)
        return self.model.lm_head(hidden)
