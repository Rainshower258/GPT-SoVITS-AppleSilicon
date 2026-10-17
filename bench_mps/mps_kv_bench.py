"""
复刻 GPT-SoVITS t2s_model.py 的解码循环结构，对比三种实现在 MPS 上的表现。
模型规格取自 configs/s1longer-v2.yaml: n_layer=24, hidden_dim=512, head=16
"""

import time

import torch
import torch.nn.functional as F

N_LAYER = 24
HIDDEN = 512
N_HEAD = 16
HEAD_DIM = HIDDEN // N_HEAD
PREFIX = 200  # 提示词长度
STEPS = 600  # 解码步数（真实上限 1500，这里取 600 以免 A 方案跑太久）
DEV = torch.device("mps")
DTYPE = torch.float32


def mem_mb():
    return torch.mps.current_allocated_memory() / 1024**2


def driver_mb():
    return torch.mps.driver_allocated_memory() / 1024**2


def bench_cat_cache(sync_every_step: bool):
    """方案 A：当前代码的实现 —— 每步 torch.cat 重新分配整个 KV cache。"""
    torch.mps.empty_cache()
    k_cache = [torch.randn(1, PREFIX, HIDDEN, device=DEV, dtype=DTYPE) for _ in range(N_LAYER)]
    v_cache = [torch.randn(1, PREFIX, HIDDEN, device=DEV, dtype=DTYPE) for _ in range(N_LAYER)]
    eos = torch.tensor(1024, device=DEV)
    torch.mps.synchronize()
    peak = mem_mb()
    t0 = time.perf_counter()
    for step in range(STEPS):
        x = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
        for i in range(N_LAYER):
            k = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
            v = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
            # ← t2s_model.py:186-187 的原样写法
            k_cache[i] = torch.cat([k_cache[i], k], dim=1)
            v_cache[i] = torch.cat([v_cache[i], v], dim=1)
            kv_len = k_cache[i].shape[1]
            q = x.view(1, 1, N_HEAD, -1).transpose(1, 2)
            kk = k_cache[i].view(1, kv_len, N_HEAD, -1).transpose(1, 2)
            vv = v_cache[i].view(1, kv_len, N_HEAD, -1).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, kk, vv)
            x = attn.transpose(1, 2).reshape(1, 1, HIDDEN)
        samples = torch.randint(0, 1024, (1, 1), device=DEV)
        if sync_every_step:
            # ← t2s_model.py:725 的 `self.EOS in samples[:, 0]`，强制 GPU→CPU 同步
            _ = bool((samples[:, 0] == eos).any())
        if step % 50 == 0:
            peak = max(peak, mem_mb())
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    peak = max(peak, mem_mb())
    del k_cache, v_cache
    return dt, peak


def bench_static_cache():
    """方案 B：预分配静态 cache（仓库里 KVCacheNHD 的做法），原地写入，不重新分配。"""
    torch.mps.empty_cache()
    max_len = PREFIX + STEPS
    k_cache = torch.zeros(N_LAYER, 1, max_len, HIDDEN, device=DEV, dtype=DTYPE)
    v_cache = torch.zeros(N_LAYER, 1, max_len, HIDDEN, device=DEV, dtype=DTYPE)
    k_cache[:, :, :PREFIX] = torch.randn(N_LAYER, 1, PREFIX, HIDDEN, device=DEV, dtype=DTYPE)
    v_cache[:, :, :PREFIX] = torch.randn(N_LAYER, 1, PREFIX, HIDDEN, device=DEV, dtype=DTYPE)
    torch.mps.synchronize()
    peak = mem_mb()
    t0 = time.perf_counter()
    for step in range(STEPS):
        pos = PREFIX + step
        x = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
        for i in range(N_LAYER):
            k = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
            v = torch.randn(1, 1, HIDDEN, device=DEV, dtype=DTYPE)
            k_cache[i, :, pos : pos + 1] = k  # 原地写入，零分配
            v_cache[i, :, pos : pos + 1] = v
            kv_len = pos + 1
            q = x.view(1, 1, N_HEAD, -1).transpose(1, 2)
            kk = k_cache[i, :, :kv_len].view(1, kv_len, N_HEAD, -1).transpose(1, 2)
            vv = v_cache[i, :, :kv_len].view(1, kv_len, N_HEAD, -1).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, kk, vv)
            x = attn.transpose(1, 2).reshape(1, 1, HIDDEN)
        _ = torch.randint(0, 1024, (1, 1), device=DEV)
        if step % 50 == 0:
            peak = max(peak, mem_mb())
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    peak = max(peak, mem_mb())
    del k_cache, v_cache
    return dt, peak


if __name__ == "__main__":
    print(f"torch {torch.__version__} | MPS {torch.backends.mps.is_available()}")
    print(f"配置: {N_LAYER}层 hidden={HIDDEN} heads={N_HEAD} prefix={PREFIX} steps={STEPS}\n")

    # 预热
    _ = torch.randn(512, 512, device=DEV) @ torch.randn(512, 512, device=DEV)
    torch.mps.synchronize()

    print("跑 A: torch.cat 动态 cache + 每步同步（= 当前代码）...")
    t_a, m_a = bench_cat_cache(sync_every_step=True)
    print(f"   {t_a:7.2f}s | {STEPS / t_a:6.2f} it/s | 峰值 {m_a:8.1f} MB | driver {driver_mb():8.1f} MB\n")

    print("跑 A': torch.cat 动态 cache，去掉每步同步...")
    t_a2, m_a2 = bench_cat_cache(sync_every_step=False)
    print(f"   {t_a2:7.2f}s | {STEPS / t_a2:6.2f} it/s | 峰值 {m_a2:8.1f} MB\n")

    print("跑 B: 预分配静态 cache（= 建议的修复）...")
    t_b, m_b = bench_static_cache()
    print(f"   {t_b:7.2f}s | {STEPS / t_b:6.2f} it/s | 峰值 {m_b:8.1f} MB\n")

    print("=" * 58)
    print(f"静态 cache 相对当前实现:  提速 {t_a / t_b:.2f}x | 峰值内存降到 {m_b / m_a * 100:.0f}%")
    print(f"每步同步单独造成的开销:  {(t_a - t_a2) / t_a2 * 100:+.1f}%")
