"""MLX 验收（二）：teacher forcing 下逐步比较 MLX 与 torch CPU fp32 的 logits。

greedy 对比（p6_verify_mlx_tokens.py）一旦某步近似平局翻转就会连锁分歧，看不出数值到底差多少。
这里把 torch 生成的 token 序列原样喂给两边，不采样，比每一步的 logits：
max|Δ| / mean|Δ| / argmax 一致率 / 不一致处的 top-2 间距。
"""
import argparse, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "GPT_SoVITS")
import torch, yaml, numpy as np
import torch.nn.functional as F
from copy import deepcopy

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="fp32:mlx_varlen,fp16:mlx_varlen,fp16:mlx_varlen:Affine,fp16:mlx_varlen:MXFP8")
ap.add_argument("--max-steps", type=int, default=400)
args = ap.parse_args()
from TTS_infer_pack.TTS import TTS, TTS_Config
preset = deepcopy(TTS_Config.default_configs["v2ProPlus"])
p = "bench_mps/_scratch_tts_config.yaml"; yaml.dump({"custom": preset}, open(p, "w"))
cfg = TTS_Config(p); cfg.device = "cpu"; cfg.t2s_device = "cpu"; cfg.is_half = False
tts = TTS(cfg)
calls = []
orig = tts.t2s_model.model.infer_panel_batch_infer
def wrapped(x, x_lens, prompts, bert_feature, **k):
    y_list, idx_list = orig(x, x_lens, prompts, bert_feature, **k)
    calls.append(dict(x=x[0].cpu(), prompts=prompts.cpu(), bert=bert_feature[0].cpu(), y=y_list[0].cpu(), idx=idx_list[0]))
    return y_list, idx_list
tts.t2s_model.model.infer_panel_batch_infer = wrapped
text = "人工智能正在改变我们的生活。从语音助手到自动驾驶，从医疗诊断到科学研究，这些技术的进步带来了巨大的便利。"
req = dict(text=text, text_lang="zh", ref_audio_path="bench_mps/ref_audio/ref_trim.wav",
           prompt_text="今天天气不错，我们去公园散步吧。", prompt_lang="zh", top_k=15, top_p=1.0,
           temperature=1.0, repetition_penalty=1.35, text_split_method="cut5",
           batch_size=1, seed=42, parallel_infer=True, streaming_mode=False, return_fragment=False)
for _ in tts.run(req): pass  # 正常采样（temperature=1），得到自然长度的 token 序列
m = tts.t2s_model.model

@torch.no_grad()
def torch_tf_logits(x, bert, prompts, tokens):
    x_item = m.ar_text_embedding(x.unsqueeze(0)) + m.bert_proj(bert.transpose(0, 1).unsqueeze(0))
    x_item = m.ar_text_position(x_item)
    y_pos = m.ar_audio_position(m.ar_audio_embedding(prompts))
    xy_pos = torch.cat([x_item, y_pos], 1)
    L, P = x_item.shape[1], prompts.shape[1]; T = L + P
    x_mask = F.pad(torch.zeros(L, L, dtype=torch.bool), (0, P), value=True)
    y_mask = F.pad(torch.triu(torch.ones(P, P, dtype=torch.bool), diagonal=1), (L, 0), value=False)
    attn = torch.cat([x_mask, y_mask], 0).view(1, T, T).unsqueeze(1).expand(-1, m.num_head, -1, -1)
    xy_dec, k_cache, v_cache = m.t2s_transformer.process_prompt(xy_pos, attn, None)
    max_T = T + 1500 + 1
    for i in range(len(k_cache)):
        kb = torch.zeros(1, max_T, k_cache[i].shape[2], dtype=k_cache[i].dtype); vb = torch.zeros_like(kb)
        kb[:, :T] = k_cache[i]; vb[:, :T] = v_cache[i]; k_cache[i], v_cache[i] = kb, vb
    kv_len = T
    out = [m.ar_predict_layer(xy_dec[:, -1])]
    for i, tok in enumerate(tokens.tolist()):
        y_emb = m.ar_audio_embedding(torch.tensor([[tok]]))
        xy_pos = y_emb * m.ar_audio_position.x_scale + m.ar_audio_position.alpha * m.ar_audio_position.pe[:, P + i]
        xy_dec, k_cache, v_cache = m.t2s_transformer.decode_next_token_static(xy_pos, k_cache, v_cache, kv_len, None)
        kv_len += 1
        out.append(m.ar_predict_layer(xy_dec[:, -1]))
    return torch.cat(out, 0)

ref = []
for c in calls:
    toks = c["y"][-c["idx"]:][: args.max_steps]
    c["tokens"] = toks
    ref.append(torch_tf_logits(c["x"], c["bert"], c["prompts"], toks))
print(f"torch CPU fp32: {len(calls)} 句, 对比步数 {[r.shape[0] for r in ref]}")

from Accel.mlx_t2s import MLXT2S
for spec in args.configs.split(","):
    parts = spec.split(":"); quant = parts[2] if len(parts) > 2 and parts[2] not in ("", "none") else None
    eng = MLXT2S(tts.configs.t2s_weights_path, backend=parts[1], dtype=parts[0], quantize=quant, compile="compile" in parts[3:])
    maxd = []; meand = []; agree = 0; total = 0; margins = []
    for c, r in zip(calls, ref):
        lg = eng.teacher_forced_logits(c["x"], c["bert"], c["prompts"], c["tokens"])
        d = (lg - r).abs()
        maxd.append(d.max().item()); meand.append(d.mean().item())
        a_r = r.argmax(-1); a_m = lg.argmax(-1)
        agree += (a_r == a_m).sum().item(); total += r.shape[0]
        top2 = r.topk(2, dim=-1).values
        margins += (top2[:, 0] - top2[:, 1])[a_r != a_m].tolist()
    ms = f"不一致处 torch top-2 间距 中位 {np.median(margins):.3f} 最大 {max(margins):.3f}" if margins else ""
    print(f"MLX {spec}: max|Δ|={max(maxd):.4f} mean|Δ|={np.mean(meand):.5f} argmax一致 {agree}/{total} ({agree/total:.1%}) {ms}")
