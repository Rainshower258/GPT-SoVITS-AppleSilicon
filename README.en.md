# GPT-SoVITS · Apple Silicon Acceleration (MPS / MLX)

<div align="center">

[**中文简体**](./README.md) | **English**

</div>

> This project is forked from [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (based on `main` at commit `48b1a01`). It is an unofficial version and is not affiliated with the upstream project. Upstream's original README is kept as [README_UPSTREAM.md](./README_UPSTREAM.md).

## About

Upstream runs inference on the CPU only on macOS. This project enables GPU inference on the Mac:

- It enables MPS and fixes the T2S decoder reallocating its KV cache with `torch.cat` at every step. That issue makes decoding slow down as sentences get longer and makes GPU memory usage grow continuously.
- It moves T2S decoding onto Apple's MLX framework.

The same workflow on upstream's own inference page runs about 3 to 4 times faster, with unchanged synthesis quality.

## Changes

| Area | Change |
|---|---|
| Device detection | Inference detects and uses MPS (`config.py`, `TTS_Config`, inference pages); MPS is excluded from training device selection |
| KV cache | The T2S decoder's KV cache is a preallocated static buffer instead of being concatenated step by step. GPU memory for long text drops from 25.5 GB to 4.9 GB |
| Attention mask | Decode steps no longer build an attention mask when there is no padding |
| Streaming synthesis | The streaming path also uses the static KV cache |
| STFT | `torch.stft` uses `return_complex=True`, replacing an interface scheduled for removal |
| Split devices | New `t2s_device` option lets T2S and VITS run on different devices |
| MLX inference | New `GPT_SoVITS/Accel/mlx_t2s.py` runs T2S decoding on MLX, in fp16 or int8 |
| MLX compilation | Decode steps are partially compiled (`mx.compile`), further reducing decode time in int8 mode |
| V5 support | v5dev / v5turbo model code, configuration and inference-page support |

The MLX model layer and V5 support are ported from upstream's `cuda_graph_accel_v5` branch. That branch's MLX code does not run on macOS as is; this project fixes the following:

- MLX's default device is set per thread, so the background thread must set it explicitly;
- the Transformer layer calls attention with arguments in the wrong order;
- it depends on an attribute that only exists in CUDA environments;
- MXFP8 quantisation uses the wrong bit width.

Upstream's scheduling layer has further tensor-shape and sampling-logic issues. Rather than patch it, this project reimplements a single-request decode loop that follows the PyTorch decoding logic, reusing only upstream's model layer.

## Speed

Test environment: Apple M5, 24 GB memory, macOS 26.3, PyTorch 2.11, MLX 0.32.

Method: each version's fast inference page (`inference_webui_fast.py`) is started and driven through its `/inference` endpoint. Reference audio, random seed and model (v2ProPlus) are identical; results are medians.

| Version | Short text (67 characters, split at punctuation) | RTF | Long text (223 characters, unsplit) | RTF |
|---|---|---|---|---|
| Upstream (CPU) | 7.50 s | 0.400 | 26.8 s | 0.494 |
| This project, MLX disabled | 3.95 s | 0.228 | 13.8 s | 0.254 |
| **This project, MLX enabled (int8 + partial compilation)** | **2.13 s** | **0.131** | **6.30 s** | **0.117** |

RTF is synthesis time divided by audio duration; lower is faster.

Measured by RTF, this is 3.1× faster than upstream on short text and 4.2× on long text. (Audio duration varies slightly between runs because of random sampling, so the ratio uses RTF rather than raw time.) On long text, the T2S stage drops from 15.4 s to 3.4 s and the VITS stage from 10.5 s to 1.27 s.

The first two rows were measured one after the other in the same session, 3 runs each. The MLX row was measured separately on an idle, fully cooled machine, 5 runs. The "MLX disabled" row was measured at the commit before MLX was introduced, which runs the same code path as the current version with `t2s_backend: torch`.

## Quality

This project only changes how the computation runs. It does not modify the model, and synthesis quality matches upstream.

- **MLX disabled**: with the same random seed, the generated speech content is identical to upstream.
- **MLX enabled**: because the computation differs, pauses and intonation for the same sentence vary slightly and naturally, to the same degree as changing the random seed in upstream. In speech-recognition tests across several voices and hundreds of sentences, the misreading rate matches upstream. Listening tests found no noise, dropped words or change in voice.
- **Precision**: int8 and fp16 sound the same; int8 is faster and recommended. MXFP8 is not recommended because its output deviates noticeably from the original.

## Installation and usage

Install as upstream describes, choosing MPS as the device:

```bash
bash install.sh --device MPS --source HF
pip install mlx
```

> Note: upstream's `--device MPS` option installs a CPU-only build of PyTorch. This project installs the standard build instead, which includes MPS support on macOS.
>
> Pretrained models are not included in this repository; download them as described upstream.

In `GPT_SoVITS/configs/tts_infer.yaml`, set the `custom` section:

```yaml
custom:
  version: v2ProPlus
  device: mps
  t2s_device: cpu          # only affects streaming synthesis
  t2s_backend: mlx         # torch or mlx
  t2s_mlx_dtype: float16
  t2s_mlx_quant: Affine    # empty for fp16; Affine for int8
  t2s_mlx_compile: true
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
```

Start it either way:

- **API server**: run `./start_api.sh`, which listens on `127.0.0.1:9880` with the same interface as upstream `api_v2.py`.
- **Web UI**: run `python webui.py`, tick "Enable Parallel Inference Version" on the inference tab, then open the inference page.

Every new option has a default. When none are set, behaviour matches upstream.

## Known limitations

- Tested only on Apple M5 so far. Other Apple Silicon chips are expected to work but have not been verified.
- The MLX engine is used for non-streaming synthesis only; streaming synthesis still uses the PyTorch implementation.
- Without reference text (reference-free mode), the PyTorch implementation is used automatically.
- The legacy inference page (opened when "Enable Parallel Inference Version" is unticked) has its own inference code. It includes only the device detection and KV cache changes and **cannot use MLX**. Use the fast inference page or the API instead.
- The VITS stage of v3 / v4 models (32-step CFM) is slow on MPS: a 20-second utterance takes more than 60 seconds. This project does not optimise it yet.
- V5 turbo runs, but its VITS stage takes about 7 times as long as v2ProPlus and occasionally has a faint electronic artifact, so it is not the default.
- Upstream's Windows portable-runtime initialisation and its CUDA Graph and FlashAttention switches are not ported; the related options on the inference pages have no effect here.
- `webui.py` no longer launches the inference pages with `python -s`, so that some layered virtual environments can find their packages. Ordinary environments are unaffected.
- The training pipeline is unchanged; training on MPS is still unsupported.

## Benchmark scripts

The benchmark and verification scripts used for testing are in `bench_mps/`:

| Script | Purpose |
|---|---|
| `e2e_baseline.py` | End-to-end synthesis benchmark with switchable device and MLX options |
| `p6_webui_compare.py` | Speed comparison through the inference page's endpoint |
| `p6_verify_mlx_logits.py` | Step-by-step comparison of MLX and PyTorch outputs |
| `verify_static_cache.py` | Numerical consistency check of the static KV cache against the original implementation |

## Acknowledgements

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) and all its contributors.
- [@XXXXRT666](https://github.com/XXXXRT666), original author of the `GPT_SoVITS/Accel/MLX` model layer.
- The Apple [MLX](https://github.com/ml-explore/mlx) team.

## License and notice

This repository is released under the MIT license, with upstream's original `LICENSE` kept as is. When synthesising any content with this project, follow the usage notice in upstream's README; users are responsible for what they synthesise.

<sub>Bug reports: devinwild48@gmail.com</sub>
