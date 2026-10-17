"""从上游 cuda_graph_accel_v5 分支（2026-10-08，`GPT_SoVITS/Accel/`）只搬了 MLX 子树。

上游的 `Accel/__init__.py` 会连带 import PyTorch 引擎、uvloop、改 cudnn/线程数等全局设置，
本 fork 只要 MLX T2S 引擎，所以这里留空，按需 `from Accel.MLX.AR.t2s_engine_mlx import T2SEngine`。
`Accel/PyTorch/AR/structs.py` 是 MLX structs 依赖的共享数据类，原样保留；`Accel/PyTorch/__init__.py` 置空。
"""
