import importlib.util

backends: list[str]

if importlib.util.find_spec("mlx") is not None:
    from .AR.sample_funcs_mlx import sample_naive_mlx
    from .AR.t2s_engine_mlx import T2SEngine as T2SEngineMLX

    backends = ["mlx_static", "mlx_varlen"]
else:
    backends = []

quantization_methods_mlx = [None, "MXFP8", "Affine"]
__all__ = ["T2SEngineMLX", "sample_naive_mlx", "backends", "quantization_methods_mlx"]
