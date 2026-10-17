"""
任意参考音频 + 任意文本的端到端说话脚本（v2ProPlus / MPS 混合部署）。

与 e2e_baseline.py 的区别：那个脚本用固定文本+固定参考音频做 md5 可复现基准，
这个脚本把 ref / prompt_text / text / 输出路径 全部参数化，用来做「拿指定音色说指定的话」，
并且把耗时拆成【模型加载】和【推理】两段单独记录，跑多次时能看出冷/热差异。

用法:
    cd <仓库根目录>
    PYTHONPATH=$(pwd) .venv/bin/python3 bench_mps/say.py \
        --ref bench_mps/ref_audio/ref_trim.wav \
        --prompt-text "今天天气不错，我们去公园散步吧。" \
        --text "要说的内容" \
        --device mps --t2s-device cpu \
        --repeat 2
"""

import argparse
import os
import resource
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "GPT_SoVITS")

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考音频路径")
    ap.add_argument("--prompt-text", required=True, help="参考音频对应的逐字文稿（必须严格对应）")
    ap.add_argument("--prompt-lang", default="zh", choices=["zh", "en", "ja", "ko", "yue", "auto"])
    ap.add_argument("--text", required=True, help="要合成的文本")
    ap.add_argument("--text-lang", default="zh", choices=["zh", "en", "ja", "ko", "yue", "auto"])
    ap.add_argument("--out", default="bench_mps/say_out.wav")
    ap.add_argument("--version", default="v2ProPlus")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    ap.add_argument("--t2s-device", default="cpu", choices=["cpu", "mps"],
                    help="混合部署：T2S 单独指定设备（当前最佳配置 mps + t2s cpu）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="推理重复次数，>1 时第一次算冷启动")
    ap.add_argument("--split", default="cut5")
    ap.add_argument("--fragment-interval", type=float, default=0.3,
                    help="段间插入的静音秒数（默认0.3）。文本被切成 N 段就有 N-1 个接缝，调小能明显减少拖沓")
    ap.add_argument("--speed-factor", type=float, default=1.0,
                    help=">1 加速。走的是时长预测缩放（sizee=codes*4/speed），不是重采样，理论上不变调")
    ap.add_argument("--keep-fragment", action="store_true")
    args = ap.parse_args()

    import torch
    import yaml
    from copy import deepcopy
    from TTS_infer_pack.TTS import TTS, TTS_Config

    # 同 e2e_baseline.py：用独立 scratch yaml，避免模型加载时的 save_configs()
    # 把 benchmark 配置写回生产文件 GPT_SoVITS/configs/tts_infer.yaml
    assert args.version in TTS_Config.default_configs, f"未知版本: {args.version}"
    preset = deepcopy(TTS_Config.default_configs[args.version])
    scratch = "bench_mps/_scratch_tts_config.yaml"
    os.makedirs(os.path.dirname(scratch) or ".", exist_ok=True)
    with open(scratch, "w") as f:
        yaml.dump({"custom": preset}, f)
    configs = TTS_Config(scratch)
    configs.device = args.device
    configs.t2s_device = args.t2s_device
    configs.is_half = False
    assert configs.version == args.version, f"版本选择失败: 期望{args.version} 实际{configs.version}"

    print(f"=== 加载模型 (device={args.device}, t2s_device={configs.t2s_device}, version={args.version}) ===")
    if args.device == "mps" or args.t2s_device == "mps":
        torch.mps.empty_cache()
    t0 = time.perf_counter()
    tts = TTS(configs)
    t_load = time.perf_counter() - t0
    print(f"模型加载耗时: {t_load:.2f}s")

    req = {
        "text": args.text,
        "text_lang": args.text_lang,
        "ref_audio_path": args.ref,
        "prompt_text": args.prompt_text,
        "prompt_lang": args.prompt_lang,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": args.split,
        "batch_size": 1,
        "seed": args.seed,
        "speed_factor": args.speed_factor,
        "fragment_interval": args.fragment_interval,
        "parallel_infer": True,
        "streaming_mode": False,
        "return_fragment": args.keep_fragment,
    }

    times = []
    for i in range(args.repeat):
        if "mps" in (args.device, args.t2s_device):
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(f"\n=== 第 {i + 1}/{args.repeat} 次推理 ===")
        t0 = time.perf_counter()
        result = None
        for sr, audio in tts.run(req):
            result = (sr, audio)
        if "mps" in (args.device, args.t2s_device):
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        sr, audio = result
        audio_len = len(audio) / sr
        print(f"  推理耗时: {dt:.2f}s   音频时长: {audio_len:.2f}s   RTF: {dt / audio_len:.3f}")

    sr, audio = result
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, audio, sr)

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    print("\n" + "=" * 56)
    print(f"参考音频 : {args.ref}")
    print(f"参考文稿 : {args.prompt_text}")
    print(f"合成文本 : {args.text}  ({len(args.text)} 字)")
    print(f"配置     : device={args.device} t2s_device={args.t2s_device} version={args.version} seed={args.seed}")
    print(f"模型加载 : {t_load:.2f}s")
    for i, dt in enumerate(times):
        tag = "冷启动" if i == 0 and args.repeat > 1 else f"第{i + 1}次"
        print(f"推理耗时 : {dt:.2f}s  ({tag})   RTF={dt / audio_len:.3f}")
    if times:
        print(f"平均推理 : {sum(times) / len(times):.2f}s")
    print(f"音频时长 : {audio_len:.2f}s  采样率={sr}")
    print(f"转写速度 : {audio_len / (sum(times) / len(times)):.2f}x 实时")
    print(f"进程峰值 RSS: {rusage.ru_maxrss / 1024 / 1024:.1f} MB")
    if args.device == "mps" or args.t2s_device == "mps":
        print(f"MPS 张量内存: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB   "
              f"驱动内存: {torch.mps.driver_allocated_memory() / 1024**2:.1f} MB")
    print(f"输出: {args.out}")
    print("=" * 56)


if __name__ == "__main__":
    main()
