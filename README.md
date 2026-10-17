# GPT-SoVITS · Apple Silicon 加速（MPS / MLX）

<div align="center">

**中文简体** | [**English**](./README.en.md)

</div>

> 本项目 fork 自 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（基于 `main` 分支提交 `48b1a01`），为非官方版本，与上游项目无隶属关系。上游原版说明见 [README_UPSTREAM.md](./README_UPSTREAM.md)。

## 关于本项目

上游版本在 macOS 上仅使用 CPU 推理。本项目为 Mac 启用 GPU 推理：

- 启用 MPS，并修复 T2S 解码中 KV cache 每一步都通过 `torch.cat` 重新分配的问题。该问题会导致解码速度随句子变长而持续下降、显存占用不断增加。
- 将 T2S 解码迁移至 Apple 的 MLX 框架执行。

在上游自带的推理页面中执行相同操作，速度提升约 3 至 4 倍，合成质量不变。

## 改动

| 类别 | 改动 |
|---|---|
| 设备识别 | 推理时识别并使用 MPS（`config.py`、`TTS_Config`、推理页）；训练设备选择不包含 MPS |
| KV cache | T2S 解码的 KV cache 改为预分配的静态缓冲区，不再逐步拼接。长文本的显存占用由 25.5 GB 降至 4.9 GB |
| 注意力掩码 | 无填充时，解码步不再构造 attention mask |
| 流式合成 | 流式合成路径同样采用静态 KV cache |
| STFT | `torch.stft` 改用 `return_complex=True`，替换即将移除的旧接口 |
| 混合设备 | 新增 `t2s_device` 配置项，T2S 与 VITS 可分别运行在不同设备上 |
| MLX 推理 | 新增 `GPT_SoVITS/Accel/mlx_t2s.py`，使用 MLX 执行 T2S 解码，支持 fp16 与 int8 |
| MLX 编译 | 对解码步进行局部编译（`mx.compile`），进一步缩短 int8 模式下的解码耗时 |
| V5 支持 | 移植 v5dev / v5turbo 的模型代码、配置与推理页支持 |

MLX 模型层与 V5 支持移植自上游的 `cuda_graph_accel_v5` 分支。该分支的 MLX 代码无法直接在 macOS 上运行，本项目修复了以下问题：

- MLX 的默认设备按线程设置，后台线程需单独指定；
- Transformer 层调用 attention 时参数顺序不匹配；
- 依赖仅 CUDA 环境才有的属性；
- MXFP8 量化的位数设置错误。

上游调度层另有若干张量维度与采样逻辑问题。本项目未直接修补，而是参照 PyTorch 版本的解码逻辑重新实现了单请求解码循环，仅复用上游的模型层。

## 速度

测试环境：Apple M5，24 GB 内存，macOS 26.3，PyTorch 2.11，MLX 0.32。

测试方法：分别启动各版本代码的快速推理页（`inference_webui_fast.py`），通过其 `/inference` 接口合成。参考音频、随机种子与模型（v2ProPlus）均相同，结果取中位数。

| 版本 | 短文本（67 字，按标点切分） | RTF | 长文本（223 字，不切分） | RTF |
|---|---|---|---|---|
| 上游原版（CPU） | 7.50 秒 | 0.400 | 26.8 秒 | 0.494 |
| 本项目，未启用 MLX | 3.95 秒 | 0.228 | 13.8 秒 | 0.254 |
| **本项目，启用 MLX（int8 + 局部编译）** | **2.13 秒** | **0.131** | **6.30 秒** | **0.117** |

RTF 为合成耗时与音频时长之比，数值越小速度越快。

按 RTF 计算，相对上游原版，短文本提速 3.1 倍，长文本提速 4.2 倍（各次合成的音频时长因随机采样略有不同，故以 RTF 而非耗时计算倍数）。以长文本为例，T2S 阶段由 15.4 秒降至 3.4 秒，VITS 阶段由 10.5 秒降至 1.27 秒。

前两行在同一轮测试中依次测得，各运行 3 次。启用 MLX 一行在设备空闲、充分降温后单独测得，运行 5 次。"未启用 MLX"一行测于引入 MLX 之前的提交，与当前版本设置 `t2s_backend: torch` 时的代码路径相同。

## 质量

本项目只改变计算方式，不修改模型，合成音质与上游一致。

- **未启用 MLX 时**：使用相同的随机种子，生成的语音内容与上游完全相同。
- **启用 MLX 时**：由于计算方式不同，同一句话的停顿和语调会有自然的细微差异，程度与上游更换随机种子时相当。在多种音色、数百句文本的语音识别测试中，读错的比例与上游持平；试听未发现杂音、吞字或音色变化。
- **精度选择**：int8 与 fp16 在听感上没有可察觉的差别，int8 速度更快，推荐使用。不建议使用 MXFP8，其输出与原版的偏差较为明显。

## 安装与使用

按上游方式安装，设备选择 MPS：

```bash
bash install.sh --device MPS --source HF
pip install mlx
```

> 注：上游的 `--device MPS` 选项实际安装的是仅支持 CPU 的 PyTorch；本项目改为安装标准版本，该版本在 macOS 上自带 MPS 支持。
>
> 本仓库不包含预训练模型，请按上游说明另行下载。

在 `GPT_SoVITS/configs/tts_infer.yaml` 的 `custom` 段中设置：

```yaml
custom:
  version: v2ProPlus
  device: mps
  t2s_device: cpu          # 仅影响流式合成
  t2s_backend: mlx         # torch 或 mlx
  t2s_mlx_dtype: float16
  t2s_mlx_quant: Affine    # 留空为 fp16；Affine 为 int8
  t2s_mlx_compile: true
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
```

可选择以下任一方式启动：

- **API 服务**：运行 `./start_api.sh`，监听 `127.0.0.1:9880`，接口与上游 `api_v2.py` 一致。
- **网页界面**：运行 `python webui.py`，在推理标签页勾选"启用并行推理版本"后打开推理页。

所有新增配置项均有默认值；未设置时，行为与上游一致。

## 已知限制

- 目前仅在 Apple M5 上完成测试。其他 Apple Silicon 芯片预期可用，尚未验证。
- MLX 引擎仅用于非流式合成；流式合成仍使用 PyTorch 实现。
- 未填写参考文本时（无参考文本模式），自动改用 PyTorch 实现。
- 旧版推理页（未勾选"启用并行推理版本"时打开的页面）使用独立的推理代码，仅包含设备识别与 KV cache 相关改动，**无法使用 MLX**。建议使用快速推理页或 API。
- v3 / v4 模型的 VITS 阶段（32 步 CFM）在 MPS 上耗时较长，一句 20 秒的音频需 60 秒以上，本项目暂未优化。
- V5 turbo 可以运行，但 VITS 阶段耗时约为 v2ProPlus 的 7 倍，且偶有轻微电子音，因此未设为默认。
- 未移植上游的 Windows 便携包初始化代码及 CUDA Graph、FlashAttention 开关，推理页上的相关选项在本项目中不生效。
- `webui.py` 启动推理页时不再使用 `python -s` 参数，以避免部分分层安装的虚拟环境找不到依赖包。常规环境不受影响。
- 训练流程未作修改，仍不支持使用 MPS 训练。

## 基准测试脚本

测试所用的基准与验证脚本位于 `bench_mps/` 目录：

| 脚本 | 用途 |
|---|---|
| `e2e_baseline.py` | 端到端合成基准，可切换设备与 MLX 选项 |
| `p6_webui_compare.py` | 通过推理页接口进行速度对比 |
| `p6_verify_mlx_logits.py` | 逐步比对 MLX 与 PyTorch 的输出 |
| `verify_static_cache.py` | 验证静态 KV cache 与原实现的数值一致性 |

## 致谢

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 及其全部贡献者。
- [@XXXXRT666](https://github.com/XXXXRT666)：`GPT_SoVITS/Accel/MLX` 模型层原作者。
- Apple [MLX](https://github.com/ml-explore/mlx) 团队。

## 许可与声明

本仓库以 MIT 许可发布，上游原 `LICENSE` 原样保留。使用本项目合成任何内容时，请遵守上游 README 中的使用声明，使用者对合成内容自行负责。

<sub>问题反馈：devinwild48@gmail.com</sub>
