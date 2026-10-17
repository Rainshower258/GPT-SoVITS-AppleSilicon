"""归因：真实 T2S 解码循环里，transformer 本体 vs 循环里其它开销 各占多少。
不改库代码——把 t2s_transformer 换成一个计时代理对象。"""
import os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "GPT_SoVITS")
import torch, yaml
from copy import deepcopy
from TTS_infer_pack.TTS import TTS, TTS_Config

preset = deepcopy(TTS_Config.default_configs["v2ProPlus"])
p = "bench_mps/_scratch_tts_config.yaml"
yaml.dump({"custom": preset}, open(p, "w"))
cfg = TTS_Config(p); cfg.device = "mps"; cfg.is_half = False
tts = TTS(cfg)

class Timed:
    def __init__(self, inner):
        self.inner = inner; self.t_prompt = 0.0; self.t_decode = 0.0; self.n = 0
    def process_prompt(self, *a, **k):
        torch.mps.synchronize(); t = time.perf_counter()
        r = self.inner.process_prompt(*a, **k)
        torch.mps.synchronize(); self.t_prompt += time.perf_counter() - t
        return r
    def decode_next_token_static(self, *a, **k):
        torch.mps.synchronize(); t = time.perf_counter()
        r = self.inner.decode_next_token_static(*a, **k)
        torch.mps.synchronize(); self.t_decode += time.perf_counter() - t; self.n += 1
        return r

proxy = Timed(tts.t2s_model.model.t2s_transformer)
tts.t2s_model.model.t2s_transformer = proxy

text = "".join(["人工智能正在改变我们生活的方方面面从语音助手到自动驾驶从医疗诊断到科学研究",
    "这些技术的进步既带来了巨大的便利也引发了广泛的讨论有人认为机器终将超越人类智慧",
    "也有人相信人类的创造力和情感是无法被算法替代的无论如何这场变革已经悄然发生",
    "并将持续深刻地影响每一个人的生活方式工作模式以及我们理解世界的方法",
    "从智能手机里的语音助手到工厂里的自动化生产线从医院的辅助诊断系统到金融领域的风险评估模型",
    "人工智能的身影几乎无处不在这既让人感到兴奋又带来了不少的担忧和思考"])
req = dict(text=text, text_lang="zh", ref_audio_path="bench_mps/ref_audio/ref_trim.wav",
           prompt_text="今天天气不错，我们去公园散步吧。", prompt_lang="zh", top_k=15, top_p=1.0,
           temperature=1.0, text_split_method="cut0", batch_size=1, seed=42,
           parallel_infer=True, streaming_mode=False, return_fragment=False)
torch.mps.synchronize(); t0 = time.perf_counter()
for _ in tts.run(req): pass
torch.mps.synchronize(); total = time.perf_counter() - t0

print("\n" + "="*56)
print(f"总推理耗时              : {total:.2f}s")
print(f"prefill(process_prompt) : {proxy.t_prompt:.2f}s")
print(f"transformer 本体 (24层) : {proxy.t_decode:.2f}s over {proxy.n} steps = {proxy.t_decode/max(proxy.n,1)*1000:.2f} ms/step")
print(f"→ 解码循环里的其它开销  : 见上面 TTS.py 打印的第3个数(T2S总耗时) 减去 transformer 本体")
print("="*56)
