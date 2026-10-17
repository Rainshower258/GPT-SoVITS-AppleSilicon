"""T2S（GPT / S1）在 MLX 上的单请求解码引擎。

为什么不用上游 `Accel/MLX/AR/t2s_engine_mlx.py` 的 T2SEngine/ModelRunner：
那套 slot 调度 + 后台线程的 runner 在 Mac 上从没跑过（2026-10-15 实测：graph_applicable 属性缺失、
MLX 流是线程局部的、take_along_axis/put_along_axis 维度不匹配、TransformerBlock 调 attention 的
签名对不上、varlen backend 缺 attn_mask/max_idx、repetition penalty 的 valid_lens 算错、
top_p 保留的是概率最小的而不是最大的……）。模型层（embedding / 位置编码 / KV cache / attention /
backends）是对的，所以只复用模型层，解码循环按本仓库 `t2s_model.py:infer_panel_batch_infer`
的语义重写一遍（bsz=1）：

- 前 11 步屏蔽 EOS（至少出 10 个 token）
- 采样顺序 = repetition penalty → top_p → temperature → top_k → 采样（与 `AR/models/utils.py:logits_to_probs` 一致）
- 停止条件 = 采样到 EOS 或 argmax 是 EOS 或超过 early_stop_num 或到 1500 步；停止时丢掉最后一个采样
- 返回的是「新生成的 token」（不含 prompt、不含被丢掉的那个），长度 = torch 路径的 idx

验收方法见 bench_mps/p6_verify_mlx_tokens.py。
"""

from __future__ import annotations

import os
import time
from importlib import import_module
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import torch

MAX_DECODE_STEPS = 1500  # 与 t2s_model.py 的 range(1500) 一致
MIN_TOKENS_BEFORE_EOS = 11  # idx < 11 时屏蔽 EOS，与 t2s_model.py 一致

_DTYPES = {"float32": mx.float32, "fp32": mx.float32, "float16": mx.float16, "fp16": mx.float16,
           "bfloat16": mx.bfloat16, "bf16": mx.bfloat16}


def _replace_key(state_dict: dict[str, torch.Tensor]):
    # 照抄上游 T2SEngine.replace_key
    out = []
    for key, value in state_dict.items():
        key = (
            key.replace("model.", "")
            .replace("in_proj_", "in_proj.")
            .replace("self_attn", "attention")
            .replace("linear", "feed_forward.linear")
            .replace("norm1", "attention_norm")
            .replace("norm2", "ffn_norm")
        )
        out.append((key, mx.array(value.to(torch.float32).cpu().numpy())))
    return out


class MLXT2S:
    def __init__(
        self,
        weights_path: os.PathLike | str,
        backend: str = "mlx_varlen",
        dtype: str = "float16",
        quantize: Optional[str] = None,  # None | "Affine"(int8 g32) | "MXFP8"(上游命名，实际是 4bit mxfp8)
        max_seq_length: int = 4096,  # prefill(文本+prompt，长句可到 600+) + 1500 步解码；varlen backend 不受这个值影响
        compile: bool = False,  # MLX 局部编译：把每层「sdpa 之后 → 下一层 qkv」这一段 mx.compile（只支持 mlx_varlen）
    ):
        self.backend = backend
        self.dtype = _DTYPES[dtype]
        self.quantize = quantize
        mx.set_default_device(mx.gpu)
        dict_s1 = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        config = dict_s1["config"]
        mod = import_module(f"Accel.MLX.AR.backends.{backend}")
        decoder = mod.T2SDecoder(config, max_seq_length=max_seq_length, max_batch_size=1)
        decoder.load_weights(_replace_key(dict_s1["weight"]))
        decoder.set_dtype(self.dtype)
        if quantize:
            decoder.quantize(quantize)
        mx.eval(decoder.parameters())
        self.decoder = decoder
        self.EOS = decoder.EOS
        self.vocab_size = decoder.vocab_size
        self.max_seq_length = max_seq_length
        self.compile = bool(compile) and backend == "mlx_varlen"
        self._segs = self._build_segments() if self.compile else None
        self.load_info = dict(backend=backend, dtype=dtype, quantize=quantize, bits=decoder.bits, group_size=decoder.group_size,
                              compile=self.compile)

    # ------------------------------------------------------------------ MLX 局部编译：局部编译
    # 每步的开销几乎全是 kernel 派发（实测前向约占每步耗时的 95%）。KV cache 的原地写在
    # mx.compile 里不生效（函数式更新又要整块拷贝），所以 cache 写入和 sdpa 保持 eager，
    # 把其余部分——out_proj → 残差+LN → FFN → 残差+LN → 下一层 in_proj+split+reshape——编译成一个
    # 图（每层一个闭包，权重当常量捕获，Linear/QuantizedLinear 都能用）。
    def _build_segments(self):
        dec = self.decoder
        layers = dec.h.layers
        H = dec.n_head

        def rs(t):  # (1,1,D) -> (1,H,1,Dh)
            return t.reshape(1, 1, H, -1).transpose(0, 2, 1, 3)

        def make_first(layer):
            def seg0(x):
                q, k, v = mx.split(layer.attention.in_proj(x), 3, -1)
                return rs(q), rs(k), rs(v)
            return mx.compile(seg0)

        def make_mid(layer, nxt):
            def seg(x, attn):  # attn: sdpa 输出 (1,H,1,Dh)
                a = layer.attention.out_proj(attn.transpose(0, 2, 1, 3).reshape(1, 1, -1))
                h = layer.attention_norm(x + a)
                out = layer.ffn_norm(h + layer.feed_forward(h))
                q, k, v = mx.split(nxt.attention.in_proj(out), 3, -1)
                return out, rs(q), rs(k), rs(v)
            return mx.compile(seg)

        def make_last(layer):
            def seg(x, attn):
                a = layer.attention.out_proj(attn.transpose(0, 2, 1, 3).reshape(1, 1, -1))
                h = layer.attention_norm(x + a)
                out = layer.ffn_norm(h + layer.feed_forward(h))
                return dec.ar_predict_layer(out[:, -1, :]).astype(mx.float32)  # 直接出 logits
            return mx.compile(seg)

        mids = [make_mid(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        return dict(first=make_first(layers[0]), mids=mids, last=make_last(layers[-1]))

    def _decode_step_compiled(self, xy_pos, input_pos, kv_cache, kv_len):
        """一步解码：返回 logits (1, V) float32。cache 写入与 sdpa 为 eager，其余编译。"""
        dec = self.decoder
        segs = self._segs
        q, k, v = segs["first"](xy_pos)
        x = xy_pos
        n = len(dec.h.layers)
        for i, layer in enumerate(dec.h.layers):
            kv_cache[i] = layer.attention.kv_class.update_cache(input_pos, k, v, kv_cache[i])  # 原地写
            k_all, v_all = kv_cache[i]
            attn = mx.fast.scaled_dot_product_attention(
                q, k_all[..., :kv_len, :], v_all[..., :kv_len, :], scale=layer.attention.scale
            )
            if i < n - 1:
                x, q, k, v = segs["mids"][i](x, attn)
            else:
                return segs["last"](x, attn)

    # ------------------------------------------------------------------ sampling (mirror of logits_to_probs)
    @staticmethod
    def _sample(logits: mx.array, previous_tokens: mx.array, top_k: int, top_p: float, temperature: float,
                repetition_penalty: float) -> tuple[mx.array, mx.array]:
        """logits (1, V) float32 → (sampled token int32 scalar, argmax int32 scalar)"""
        if repetition_penalty != 1.0:
            score = mx.take_along_axis(logits, previous_tokens, axis=1)
            score = mx.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits = mx.put_along_axis(logits, previous_tokens, score, axis=1)
        # torch 的 logits_to_probs 用 scatter_ 原地改 logits，之后 infer_panel 里的
        # torch.argmax(logits) 看到的是「加过 repetition penalty 的」logits——这里保持一致
        argmax_token = mx.argmax(logits, axis=-1)
        if top_p is not None and top_p < 1.0:
            sorted_indices = mx.argsort(-logits, axis=-1)
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            remove_sorted = cum_probs > top_p
            remove_sorted[:, 0] = False  # 至少保留概率最大的一个（torch 版语义）
            remove = mx.put_along_axis(mx.zeros_like(remove_sorted), sorted_indices, remove_sorted, axis=1)
            logits = mx.where(remove, -mx.inf, logits)
        greedy = temperature <= 1e-5
        logits = logits / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.shape[-1])
            pivot = mx.sort(logits, axis=-1)[:, -k : -k + 1] if k < logits.shape[-1] else mx.min(logits, axis=-1, keepdims=True)
            logits = mx.where(logits < pivot, -mx.inf, logits)
        if greedy:
            token = mx.argmax(logits, axis=-1)
        else:
            token = mx.random.categorical(logits, axis=-1)
        return token.astype(mx.int32), argmax_token.astype(mx.int32)

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def infer(
        self,
        phones: torch.Tensor,  # (L,) long
        bert: torch.Tensor,  # (1024, L)
        prompt: torch.Tensor,  # (1, P) long semantic prompt tokens
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        early_stop_num: int = -1,
    ) -> tuple[torch.Tensor, int, dict]:
        """返回 (生成的 token 1-D LongTensor (CPU), 生成步数 idx, 统计信息)"""
        dec = self.decoder
        t0 = time.perf_counter()
        x = [mx.array(phones.detach().cpu().numpy()).astype(mx.int32)]
        y = mx.array(prompt.detach().cpu().numpy()).astype(mx.int32).reshape(1, -1)
        bert_mx = [mx.array(bert.detach().cpu().to(torch.float32).numpy()).astype(self.dtype)]
        L = x[0].shape[0]
        P = y.shape[1]
        T = L + P
        max_steps = min(MAX_DECODE_STEPS, self.max_seq_length - T - 1)
        if max_steps < 1:
            raise ValueError(f"prefill 长度 {T} 已超过 max_seq_length={self.max_seq_length}")

        # ---- prefill：文本行只看文本，prompt 行看全部文本 + 因果 prompt（同 torch 的 xy_attn_mask）
        attn_mask = mx.zeros((1, 1, T, T), dtype=mx.bool_)
        attn_mask[:, :, :, :L] = True
        attn_mask[:, :, L:, L:] = ~mx.triu(mx.ones((P, P), dtype=mx.bool_), k=1)
        kv_cache = dec.init_cache(1)
        xy_pos = dec.embed(x, y, bert_mx)  # (1, T, D)
        xy_dec = dec.h.prefill(xy_pos, kv_cache, attn_mask)
        hidden = xy_dec[:, -1:, :]
        mx.eval(hidden, kv_cache)
        t_prefill = time.perf_counter() - t0

        # ---- decode
        y_buf = mx.full((1, self.max_seq_length), self.EOS, dtype=mx.int32)
        y_buf[:, :P] = y
        kv_len = T  # 已写入 KV cache 的长度
        generated: list[int] = []
        idx = 0
        stop_reason = "max_steps"
        logits = dec.ar_predict_layer(hidden[:, -1, :]).astype(mx.float32)  # (1, V)，第 0 步来自 prefill
        for idx in range(max_steps):
            if idx < MIN_TOKENS_BEFORE_EOS:
                logits = logits[:, :-1]
            token, argmax_token = self._sample(
                logits, y_buf[:, : P + idx], top_k, top_p, temperature, repetition_penalty
            )
            tok, am = mx.stack([token, argmax_token]).tolist()  # 一次同步拿两个值
            tok, am = int(tok[0]), int(am[0])
            n_new = idx + 1  # 含本步 → 对应 torch 里 concat 后的 y.shape[1] - prefix_len
            if tok == self.EOS or am == self.EOS:
                stop_reason = "eos"
                break  # 与 torch 一致：丢掉本步采样
            if (early_stop_num != -1 and n_new > early_stop_num) or idx == max_steps - 1:
                stop_reason = "early_stop" if idx != max_steps - 1 else "max_steps"
                break  # torch 也是 y[:, :-1]
            generated.append(tok)
            y_buf[:, P + idx] = token
            # 下一步输入：音频 embedding + 位置 pe[P + idx]（ar_audio_position 用的是 input_pos - 1）
            y_emb = dec.ar_audio_embedding(token.reshape(1, 1))
            xy_pos = dec.ar_audio_position(mx.array([P + idx + 1], dtype=mx.int32), y_emb)
            kv_len += 1
            input_pos = mx.array([kv_len], dtype=mx.int32)
            if self.compile:
                logits = self._decode_step_compiled(xy_pos, input_pos, kv_cache, kv_len)
                continue
            if self.backend == "mlx_varlen":
                kw = dict(max_idx=kv_len, attn_mask=mx.ones((1, 1, 1, kv_len), dtype=mx.bool_))
            else:  # mlx_static：mask 覆盖整个 max_seq_length，已写入的位置为 True
                m = mx.zeros((1, 1, 1, self.max_seq_length), dtype=mx.bool_)
                m[..., :kv_len] = True
                kw = dict(max_idx=kv_len, attn_mask=m)
            hidden = dec.h(xy_pos, input_pos, kv_cache, **kw)
            logits = dec.ar_predict_layer(hidden[:, -1, :]).astype(mx.float32)
        n_gen = len(generated)
        if n_gen == 0:
            generated = [0]  # torch: "bad zero prediction"
        t_total = time.perf_counter() - t0
        stats = dict(prefill_s=t_prefill, total_s=t_total, steps=idx + 1, n_gen=n_gen,
                     tok_per_s=(idx + 1) / max(t_total - t_prefill, 1e-6), stop=stop_reason)
        return torch.tensor(generated, dtype=torch.long), n_gen, stats


    # ------------------------------------------------------------------ verification helper
    @torch.no_grad()
    def teacher_forced_logits(self, phones: torch.Tensor, bert: torch.Tensor, prompt: torch.Tensor,
                              tokens: torch.Tensor) -> torch.Tensor:
        """喂给定 token 序列（不采样），返回每一步的原始 logits (n+1, V) float32，用来和 torch 逐步比数值。"""
        dec = self.decoder
        x = [mx.array(phones.detach().cpu().numpy()).astype(mx.int32)]
        y = mx.array(prompt.detach().cpu().numpy()).astype(mx.int32).reshape(1, -1)
        bert_mx = [mx.array(bert.detach().cpu().to(torch.float32).numpy()).astype(self.dtype)]
        L, P = x[0].shape[0], y.shape[1]
        T = L + P
        attn_mask = mx.zeros((1, 1, T, T), dtype=mx.bool_)
        attn_mask[:, :, :, :L] = True
        attn_mask[:, :, L:, L:] = ~mx.triu(mx.ones((P, P), dtype=mx.bool_), k=1)
        kv_cache = dec.init_cache(1)
        xy_dec = dec.h.prefill(dec.embed(x, y, bert_mx), kv_cache, attn_mask)
        hidden = xy_dec[:, -1:, :]
        out = [dec.ar_predict_layer(hidden[:, -1, :]).astype(mx.float32)]
        kv_len = T
        for i, tok in enumerate(tokens.tolist()):
            token = mx.array([tok], dtype=mx.int32)
            y_emb = dec.ar_audio_embedding(token.reshape(1, 1))
            xy_pos = dec.ar_audio_position(mx.array([P + i + 1], dtype=mx.int32), y_emb)
            kv_len += 1
            if self.compile:
                out.append(self._decode_step_compiled(xy_pos, mx.array([kv_len], dtype=mx.int32), kv_cache, kv_len))
                continue
            if self.backend == "mlx_varlen":
                kw = dict(max_idx=kv_len, attn_mask=mx.ones((1, 1, 1, kv_len), dtype=mx.bool_))
            else:
                m = mx.zeros((1, 1, 1, self.max_seq_length), dtype=mx.bool_)
                m[..., :kv_len] = True
                kw = dict(max_idx=kv_len, attn_mask=m)
            hidden = dec.h(xy_pos, mx.array([kv_len], dtype=mx.int32), kv_cache, **kw)
            out.append(dec.ar_predict_layer(hidden[:, -1, :]).astype(mx.float32))
        res = mx.concatenate(out, axis=0)
        mx.eval(res)
        return torch.from_numpy(__import__("numpy").array(res))
