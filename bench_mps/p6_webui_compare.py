"""以上游前端（inference_webui_fast.py 的 /inference 接口）为口径的速度对比驱动。

对一个已经起好的快速推理页：预热 1 次短文本，然后短文本 × N、长文本 × N，
记录每次的墙上时间、音频时长、RTF，并从服务端日志里抓 TTS.py 打印的四段计时
（参考音频 / 文本前端 / T2S / VITS）。输出 JSON 到 --out。
"""
import argparse, json, re, sys, time, statistics as st
import soundfile as sf
from gradio_client import Client, handle_file

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--log", required=True, help="服务端 stdout 日志，用来抓四段计时")
ap.add_argument("--out", required=True)
ap.add_argument("--repeat", type=int, default=3)
a = ap.parse_args()

import os
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_audio", "ref_trim.wav")  # 自备 3–10s 参考音频放这里（gitignore）
PROMPT = "今天天气不错，我们去公园散步吧。"
SHORT = "人工智能正在改变我们的生活。从语音助手到自动驾驶，从医疗诊断到科学研究，这些技术的进步带来了巨大的便利。"
LONG = "".join(["人工智能正在改变我们生活的方方面面从语音助手到自动驾驶从医疗诊断到科学研究",
    "这些技术的进步既带来了巨大的便利也引发了广泛的讨论有人认为机器终将超越人类智慧",
    "也有人相信人类的创造力和情感是无法被算法替代的无论如何这场变革已经悄然发生",
    "并将持续深刻地影响每一个人的生活方式工作模式以及我们理解世界的方法",
    "从智能手机里的语音助手到工厂里的自动化生产线从医院的辅助诊断系统到金融领域的风险评估模型",
    "人工智能的身影几乎无处不在这既让人感到兴奋又带来了不少的担忧和思考"])

c = Client(f"http://127.0.0.1:{a.port}/", verbose=False)
info = c.view_api(return_format="dict", print_info=False)
names = [p["parameter_name"] for p in info["named_endpoints"]["/inference"]["parameters"]]
ref = handle_file(REF)

def call(text, cut):
    kw = dict(text=text, text_lang="中英混合", ref_audio_path=ref, aux_ref_audio_paths=None, prompt_text=PROMPT, prompt_lang="中文",
              top_k=15, top_p=1.0, temperature=1.0, text_split_method=cut, batch_size=1, speed_factor=1.0, ref_text_free=False,
              split_bucket=True, fragment_interval=0.3, seed=42, keep_random=False, parallel_infer=True, repetition_penalty=1.35,
              sample_steps="32", super_sampling=False, use_cuda_graph=False, use_flash_attention=False, cfg_rate=0.0)
    kw = {k: v for k, v in kw.items() if k in names}
    t = time.perf_counter(); out = c.predict(**kw, api_name="/inference"); wall = time.perf_counter() - t
    d, sr = sf.read(out[0] if isinstance(out, (list, tuple)) else out)
    return wall, len(d) / sr

def log_timings():
    rows = []
    for line in open(a.log, errors="ignore"):
        m = re.match(r"^(\d+\.\d+)\t(\d+\.\d+)\t(\d+\.\d+)\t(\d+\.\d+)\s*$", line)
        if m: rows.append([float(x) for x in m.groups()])
    return rows

res = {"label": a.label, "param_names": names}
call(SHORT, "按标点符号切")  # 预热
n0 = len(log_timings())
for key, text, cut in [("short", SHORT, "按标点符号切"), ("long", LONG, "不切")]:
    runs = []
    for i in range(a.repeat):
        wall, dur = call(text, cut)
        runs.append(dict(wall=wall, dur=dur, rtf=wall / dur))
        print(f"[{a.label}] {key} #{i+1}: wall {wall:.2f}s audio {dur:.2f}s RTF {wall/dur:.3f}", flush=True)
    tl = log_timings()[n0:]; n0 = len(log_timings())
    seg = None
    if len(tl) == a.repeat:
        seg = dict(ref=st.median(r[0] for r in tl), text=st.median(r[1] for r in tl), t2s=st.median(r[2] for r in tl), vits=st.median(r[3] for r in tl))
    res[key] = dict(runs=runs, wall_med=st.median(r["wall"] for r in runs), dur_med=st.median(r["dur"] for r in runs),
                    rtf_med=st.median(r["rtf"] for r in runs), segments=seg)
    print(f"[{a.label}] {key} 中位: wall {res[key]['wall_med']:.2f}s RTF {res[key]['rtf_med']:.3f} 分段 {seg}", flush=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
