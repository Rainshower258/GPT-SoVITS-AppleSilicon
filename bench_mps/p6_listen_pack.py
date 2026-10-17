"""听样：同一参考音色，6 句 × 4 个版本，按句分文件夹，文件名带顺序号，方便逐个停下来听。
用法: PYTHONPATH=$(pwd):$(pwd)/GPT_SoVITS .venv/bin/python bench_mps/p6_listen_pack.py <ref.wav> "<prompt_text>" <prompt_lang> <out_dir>
"""
import os, sys, csv, gc, time
sys.path.insert(0, "."); sys.path.insert(0, "GPT_SoVITS")
import torch, yaml, numpy as np, soundfile as sf
from copy import deepcopy
from TTS_infer_pack.TTS import TTS, TTS_Config

ref, prompt_text, prompt_lang, out_dir = sys.argv[1:5]
SENTENCES_TSV = os.environ.get("SENTENCES_TSV", "eval/sentences.tsv")  # 句子表（TSV：id, type, terms, text）
sents = {r["id"]: r["text"] for r in csv.DictReader(open(SENTENCES_TSV), delimiter="\t")}
ids = ["z001", "z010", "z020", "m001", "m010", "m050"]
versions = [  # (序号_名字, version, overrides)
    ("1_torch基线", "v2ProPlus", dict(t2s_backend="torch", t2s_device="cpu")),
    ("2_mlx_fp16", "v2ProPlus", dict(t2s_backend="mlx", t2s_mlx_dtype="float16", t2s_mlx_quant=None)),
    ("3_v5turbo", "v5turbo", dict(t2s_backend="mlx", t2s_mlx_dtype="float16", t2s_mlx_quant=None)),
    ("4_mlx_int8", "v2ProPlus", dict(t2s_backend="mlx", t2s_mlx_dtype="float16", t2s_mlx_quant="Affine")),
]
for sid in ids:
    os.makedirs(os.path.join(out_dir, f"{sid} {sents[sid][:18]}"), exist_ok=True)
for name, version, ov in versions:
    preset = deepcopy(TTS_Config.default_configs[version])
    p = "bench_mps/_scratch_tts_config.yaml"; yaml.dump({"custom": preset}, open(p, "w"))
    cfg = TTS_Config(p); cfg.device = "mps"; cfg.is_half = False
    cfg.t2s_device = torch.device(ov.get("t2s_device", "mps"))
    cfg.t2s_backend = ov["t2s_backend"]; cfg.t2s_mlx_dtype = ov.get("t2s_mlx_dtype", "float16"); cfg.t2s_mlx_quant = ov.get("t2s_mlx_quant"); cfg.t2s_mlx_backend = "mlx_varlen"
    tts = TTS(cfg)
    for sid in ids:
        req = dict(text=sents[sid], text_lang="zh", ref_audio_path=ref, prompt_text=prompt_text, prompt_lang=prompt_lang,
                   top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35, text_split_method="cut5",
                   batch_size=1, seed=42, parallel_infer=True, streaming_mode=False, return_fragment=False)
        t = time.perf_counter()
        for sr, audio in tts.run(req):
            pass
        out = os.path.join(out_dir, f"{sid} {sents[sid][:18]}", f"{sid}_{name}.wav")
        sf.write(out, audio, sr)
        print(f"[{name}] {sid} {len(audio)/sr:.1f}s audio in {time.perf_counter()-t:.1f}s -> {out}", flush=True)
    del tts; gc.collect(); torch.mps.empty_cache()
print("PACKDONE")
