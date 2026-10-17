"""混合设备验收：T2S在CPU上跑，token是否与纯CPU配置逐位相同（不受VITS设备影响）。"""
import os, sys, hashlib
sys.path.insert(0, "."); sys.path.insert(0, "GPT_SoVITS")
import torch, yaml
from copy import deepcopy
from TTS_infer_pack.TTS import TTS, TTS_Config

def run(device, t2s_device):
    preset = deepcopy(TTS_Config.default_configs["v2ProPlus"])
    p = "bench_mps/_scratch_tts_config.yaml"
    yaml.dump({"custom": preset}, open(p, "w"))
    cfg = TTS_Config(p)
    cfg.device = device
    cfg.t2s_device = t2s_device
    cfg.is_half = False
    tts = TTS(cfg)

    captured = {}
    orig = tts.t2s_model.model.infer_panel_batch_infer
    def wrapped(*a, **k):
        y_list, idx_list = orig(*a, **k)
        captured["tokens"] = [y.cpu() for y in y_list]
        captured["idx"] = idx_list
        return y_list, idx_list
    tts.t2s_model.model.infer_panel_batch_infer = wrapped

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
    for _ in tts.run(req): pass
    return captured

ref = run("cpu", "cpu")
hyb = run("mps", "cpu")

t_ref = torch.cat(ref["tokens"])
t_hyb = torch.cat(hyb["tokens"])
print(f"纯CPU  : shape={tuple(t_ref.shape)} idx={ref['idx']}")
print(f"混合   : shape={tuple(t_hyb.shape)} idx={hyb['idx']}")
identical = torch.equal(t_ref, t_hyb)
print(f"token 逐位相同: {identical}")
if not identical:
    diff = (t_ref.long() - t_hyb.long()).abs()
    print(f"  第一个不同位置: {torch.nonzero(diff)[:5]}")
