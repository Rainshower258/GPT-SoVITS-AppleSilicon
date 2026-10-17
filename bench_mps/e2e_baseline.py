"""
端到端基线测试：固定文本 + 固定参考音频，跑一次真实推理。
记录：总耗时、T2S it/s（从 tqdm stderr 抓不到就手动计时 run() 内部）、峰值内存。

用法:
    cd <仓库根目录>
    PYTHONPATH=$(pwd) .venv/bin/python3 bench_mps/e2e_baseline.py --device cpu
    PYTHONPATH=$(pwd) .venv/bin/python3 bench_mps/e2e_baseline.py --device mps   # 启用 MPS 后
"""

import argparse
import resource
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "GPT_SoVITS")

import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--version", default="v2ProPlus")  # v2的声码器较老，容易有金属声/电音；v2ProPlus音质更干净且是webui.py的实际默认版本
    ap.add_argument("--long", action="store_true", help="用长文本触发多轮长序列解码，复现KV cache churn")
    ap.add_argument(
        "--streaming", action="store_true",
        help="parallel_infer=False+streaming_mode=True，走 infer_panel_naive 而不是 infer_panel_batch_infer"
        "（流式路径；api_v2.py 的 streaming_mode=2/3 就是这条）",
    )
    ap.add_argument(
        "--t2s-device", default=None, choices=["cpu", "mps"],
        help="混合部署：T2S 单独指定设备，不传则跟 --device 一致。"
        "典型用法 --device mps --t2s-device cpu，VITS走MPS、T2S走CPU",
    )
    ap.add_argument("--t2s-backend", default="torch", choices=["torch", "mlx"],
                    help="MLX：T2S 用 MLX 引擎（Apple GPU）还是 torch")
    ap.add_argument("--mlx-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--mlx-quant", default=None, choices=[None, "Affine", "MXFP8"])
    ap.add_argument("--mlx-backend", default="mlx_varlen", choices=["mlx_varlen", "mlx_static"])
    ap.add_argument("--mlx-compile", action="store_true", help="MLX 局部编译：局部 mx.compile 的解码步")
    args = ap.parse_args()

    import os
    import torch
    import yaml
    from copy import deepcopy

    from TTS_infer_pack.TTS import TTS, TTS_Config

    # 注意两个坑，都会导致「看起来选对了版本，实际没有」：
    # 1) TTS_Config(args.version) 是错的——它会把字符串当"配置文件路径"，文件不存在时
    #    静默 fallback 到硬编码默认值（永远是v2的custom配置，不管传什么字符串）。
    # 2) 用 TTS_Config({"custom": preset}) 传字典虽然能正确选中版本，但 self.configs_path
    #    会保持类初始化时的默认值 = 生产配置 GPT_SoVITS/configs/tts_infer.yaml；而模型加载过程
    #    (init_t2s_weights/init_vits_weights) 会自动调用 save_configs() 写回 configs_path——
    #    这会把 benchmark 跑的配置覆盖进生产文件，污染 api_v2.py 实际会读到的默认配置。
    # 正确做法：用一个独立的 scratch yaml 路径（字符串），走 TTS_Config 设计好的"文件路径"分支，
    # 这样任何自动 save 都只会写回这个 scratch 文件，不碰生产配置。
    assert args.version in TTS_Config.default_configs, f"未知版本: {args.version}"
    preset = deepcopy(TTS_Config.default_configs[args.version])
    scratch_config_path = "bench_mps/_scratch_tts_config.yaml"
    os.makedirs(os.path.dirname(scratch_config_path) or ".", exist_ok=True)
    with open(scratch_config_path, "w") as f:
        yaml.dump({"custom": preset}, f)
    configs = TTS_Config(scratch_config_path)
    configs.device = args.device
    # configs.device 在 __init__ 之后手动覆盖，t2s_device 不会跟着联动
    # （生产环境走 yaml 走正常 __init__ 流程不会有这个问题，这里是 benchmark 脚本自己的坑）
    configs.t2s_device = args.t2s_device if args.t2s_device else configs.device
    configs.is_half = False  # 固定 fp32，避免与精度变量混在一起
    configs.t2s_backend = args.t2s_backend
    configs.t2s_mlx_dtype = args.mlx_dtype
    configs.t2s_mlx_quant = args.mlx_quant
    configs.t2s_mlx_backend = args.mlx_backend
    configs.t2s_mlx_compile = args.mlx_compile
    assert configs.version == args.version, f"版本选择失败: 期望{args.version}实际{configs.version}"

    print(f"=== 加载模型 (device={args.device}, t2s_device={configs.t2s_device}, version={args.version}) ===")
    t_load0 = time.perf_counter()
    tts = TTS(configs)
    t_load1 = time.perf_counter()
    print(f"模型加载耗时: {t_load1 - t_load0:.2f}s")

    ref_audio = "bench_mps/ref_audio/ref_trim.wav"  # 裁到第一句话的自然停顿处(4.85s)，文字和音频严格对应
    prompt_text = "今天天气不错，我们去公园散步吧。"
    if args.long:
        # 拉长句子（去掉切分符号让 text_split_method 少切，单句decode步数更高，逼近1500步上限）
        text = "".join(
            [
                "人工智能正在改变我们生活的方方面面从语音助手到自动驾驶从医疗诊断到科学研究",
                "这些技术的进步既带来了巨大的便利也引发了广泛的讨论有人认为机器终将超越人类智慧",
                "也有人相信人类的创造力和情感是无法被算法替代的无论如何这场变革已经悄然发生",
                "并将持续深刻地影响每一个人的生活方式工作模式以及我们理解世界的方法",
                "从智能手机里的语音助手到工厂里的自动化生产线从医院的辅助诊断系统到金融领域的风险评估模型",
                "人工智能的身影几乎无处不在这既让人感到兴奋又带来了不少的担忧和思考",
            ]
        )
    else:
        text = (
            "人工智能正在改变我们生活的方方面面，从语音助手到自动驾驶，"
            "从医疗诊断到科学研究，这些技术的进步既带来了巨大的便利，也引发了广泛的讨论。"
        )

    req = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": ref_audio,
        "prompt_text": prompt_text,
        "prompt_lang": "zh",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0" if args.long else "cut5",  # cut0=不切分，逼近单次1500步上限
        "batch_size": 1,
        "seed": 42,
        "parallel_infer": not args.streaming,
        "streaming_mode": args.streaming,
        "return_fragment": False,
    }

    print("=== 开始推理 ===")
    if args.device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    if args.streaming:
        # streaming_mode 下 tts.run() 会多次 yield 分段音频（每段之间已含静音间隔），
        # 拼接成完整音频才能和非流式的输出做时长/md5对照
        import numpy as np
        chunks = []
        sr = None
        for sr, audio in tts.run(req):
            chunks.append(audio)
        audio = np.concatenate(chunks, axis=0)
        result = (sr, audio)
    else:
        result = None
        for sr, audio in tts.run(req):
            result = (sr, audio)
    if args.device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    sr, audio = result
    hybrid_tag = f"_t2s-{args.t2s_device}" if args.t2s_device and args.t2s_device != args.device else ""
    if args.t2s_backend == "mlx":
        hybrid_tag = f"_t2s-mlx-{args.mlx_dtype}" + (f"-{args.mlx_quant}" if args.mlx_quant else "") + ("-static" if args.mlx_backend == "mlx_static" else "") + ("-compiled" if args.mlx_compile else "")
    tag = f"{args.device}_{args.version}{hybrid_tag}{'_long' if args.long else ''}{'_streaming' if args.streaming else ''}"
    out_path = f"bench_mps/e2e_output_{tag}.wav"
    sf.write(out_path, audio, sr)

    dt = t1 - t0
    audio_len_s = len(audio) / sr
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = rusage.ru_maxrss / 1024 / 1024  # macOS: bytes -> MB

    print("\n" + "=" * 50)
    print(f"设备: {args.device}")
    print(f"输入文本长度: {len(text)} 字")
    print(f"生成音频时长: {audio_len_s:.2f}s")
    print(f"推理耗时: {dt:.2f}s")
    print(f"RTF (推理耗时/音频时长): {dt / audio_len_s:.3f}")
    print(f"进程峰值 RSS: {peak_rss_mb:.1f} MB")
    if args.device == "mps":
        print(f"MPS 张量内存 (当前): {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
        print(f"MPS 驱动内存 (当前): {torch.mps.driver_allocated_memory() / 1024**2:.1f} MB")
    print(f"输出已保存: {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
