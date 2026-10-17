"""MLX 验收（一）：MLX T2S 引擎 vs torch CPU fp32 的 greedy token 序列对比。

思路：temperature=1e-5 时两边都退化成 argmax（torch: logits/1e-5 后 softmax→multinomial；
MLX: 直接 argmax），采样不再引入随机性，token 序列就成了二值可比的东西。
先跑一遍 torch CPU fp32，把 infer_panel_batch_infer 每次调用的真实输入（phones/bert/prompt）
和输出 token 都截下来；再把同样的输入喂给 MLX（fp32 / fp16 / 量化），比较 token 序列。

用法:
    PYTHONPATH=$(pwd):$(pwd)/GPT_SoVITS .venv/bin/python bench_mps/p6_verify_mlx_tokens.py [--long]
"""
import argparse, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "GPT_SoVITS")
import torch, yaml
from copy import deepcopy

ap = argparse.ArgumentParser()
ap.add_argument("--long", action="store_true")
ap.add_argument("--configs", default="fp32:mlx_varlen,fp16:mlx_varlen,fp16:mlx_static,fp16:mlx_varlen:Affine,fp16:mlx_varlen:MXFP8")
args = ap.parse_args()

from TTS_infer_pack.TTS import TTS, TTS_Config

preset = deepcopy(TTS_Config.default_configs["v2ProPlus"])
p = "bench_mps/_scratch_tts_config.yaml"
yaml.dump({"custom": preset}, open(p, "w"))
cfg = TTS_Config(p)
cfg.device = "cpu"; cfg.t2s_device = "cpu"; cfg.is_half = False
tts = TTS(cfg)

calls = []
orig = tts.t2s_model.model.infer_panel_batch_infer
def wrapped(x, x_lens, prompts, bert_feature, **k):
    t = time.perf_counter()
    y_list, idx_list = orig(x, x_lens, prompts, bert_feature, **k)
    calls.append(dict(x=[i.cpu() for i in x], x_lens=x_lens.cpu(), prompts=prompts.cpu(),
                      bert=[b.cpu() for b in bert_feature], y=[y.cpu() for y in y_list], idx=list(idx_list),
                      secs=time.perf_counter() - t))
    return y_list, idx_list
tts.t2s_model.model.infer_panel_batch_infer = wrapped

if args.long:
    text = "".join(["人工智能正在改变我们生活的方方面面从语音助手到自动驾驶从医疗诊断到科学研究",
        "这些技术的进步既带来了巨大的便利也引发了广泛的讨论有人认为机器终将超越人类智慧",
        "也有人相信人类的创造力和情感是无法被算法替代的无论如何这场变革已经悄然发生",
        "并将持续深刻地影响每一个人的生活方式工作模式以及我们理解世界的方法",
        "从智能手机里的语音助手到工厂里的自动化生产线从医院的辅助诊断系统到金融领域的风险评估模型",
        "人工智能的身影几乎无处不在这既让人感到兴奋又带来了不少的担忧和思考"])
else:
    text = "人工智能正在改变我们的生活。从语音助手到自动驾驶，从医疗诊断到科学研究，这些技术的进步带来了巨大的便利。"
req = dict(text=text, text_lang="zh", ref_audio_path="bench_mps/ref_audio/ref_trim.wav",
           prompt_text="今天天气不错，我们去公园散步吧。", prompt_lang="zh", top_k=15, top_p=1.0,
           temperature=1e-5, repetition_penalty=1.35, text_split_method="cut0" if args.long else "cut5",
           batch_size=1, seed=42, parallel_infer=True, streaming_mode=False, return_fragment=False)
for _ in tts.run(req): pass
torch_tokens = []
for c in calls:
    assert len(c["x"]) == 1, "本脚本只验 bsz=1"
    y, idx = c["y"][0], c["idx"][0]
    torch_tokens.append(y[-idx:] if idx > 0 else y[:0])
print(f"torch CPU fp32 greedy: {len(calls)} 句, token 数 {[t.numel() for t in torch_tokens]}, T2S 总耗时 {sum(c['secs'] for c in calls):.2f}s")

from Accel.mlx_t2s import MLXT2S
early_stop = tts.configs.hz * tts.configs.max_sec
for spec in args.configs.split(","):
    parts = spec.split(":")
    dtype, backend = parts[0], parts[1]
    quant = parts[2] if len(parts) > 2 else None
    eng = MLXT2S(tts.configs.t2s_weights_path, backend=backend, dtype=dtype, quantize=quant)
    # 预热一次（编译/首次分配），不计时
    c = calls[0]
    eng.infer(c["x"][0], c["bert"][0], c["prompts"], top_k=15, top_p=1.0, temperature=1e-5, repetition_penalty=1.35, early_stop_num=early_stop)
    total = 0.0; n_same = 0; report = []
    for c, ref in zip(calls, torch_tokens):
        tok, n, st = eng.infer(c["x"][0], c["bert"][0], c["prompts"], top_k=15, top_p=1.0, temperature=1e-5,
                               repetition_penalty=1.35, early_stop_num=early_stop)
        total += st["total_s"]
        same = torch.equal(tok, ref)
        n_same += same
        if same:
            report.append(f"    句 len={ref.numel()} 完全一致  {st['tok_per_s']:.0f} tok/s")
        else:
            m = min(tok.numel(), ref.numel())
            first = int(torch.nonzero(tok[:m] != ref[:m])[0]) if m and bool((tok[:m] != ref[:m]).any()) else m
            report.append(f"    句 len torch={ref.numel()} mlx={tok.numel()} 首个分歧位置={first} ({first/max(ref.numel(),1):.0%})  {st['tok_per_s']:.0f} tok/s  stop={st['stop']}")
    print(f"MLX {spec} ({eng.load_info}): {n_same}/{len(calls)} 句 token 逐位相同, T2S 总耗时 {total:.2f}s (torch CPU {sum(c['secs'] for c in calls):.2f}s)")
    print("\n".join(report))
