"""验证静态KV cache方案的三个技术假设（不碰真实模型，纯机制验证）:
  A. @torch.jit.script 的类里能否用 narrow+copy_ 原地写、能否 view/transpose 出 SDPA 要的形状
  B. 切片 view 是否零拷贝（storage 指针相同）
  C. MPS 上 SDPA 能否吃非连续的静态 buffer 切片，且数值与 torch.cat 版本一致
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
H, Dh, B = 16, 32, 1
D = H * Dh


# ---------- A. TorchScript 能否编译静态写入 ----------
@torch.jit.script
class StaticBlock:
    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    # 布局1: (B, T, D) 存储，每步 view 成 (B,H,T,d)
    def step_btd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 k_buf: torch.Tensor, v_buf: torch.Tensor, pos: int):
        k_buf.narrow(1, pos, 1).copy_(k)
        v_buf.narrow(1, pos, 1).copy_(v)
        kv_len = pos + 1
        bsz = q.shape[0]
        qq = q.view(bsz, q.shape[1], self.num_heads, -1).transpose(1, 2)
        kk = k_buf.narrow(1, 0, kv_len).view(bsz, kv_len, self.num_heads, -1).transpose(1, 2)
        vv = v_buf.narrow(1, 0, kv_len).view(bsz, kv_len, self.num_heads, -1).transpose(1, 2)
        return F.scaled_dot_product_attention(qq, kk, vv)

    # 布局2: (B, H, T, d) 存储，免 view
    def step_bhtd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  k_buf: torch.Tensor, v_buf: torch.Tensor, pos: int):
        k_buf.narrow(2, pos, 1).copy_(k)
        v_buf.narrow(2, pos, 1).copy_(v)
        kv_len = pos + 1
        kk = k_buf.narrow(2, 0, kv_len)
        vv = v_buf.narrow(2, 0, kv_len)
        return F.scaled_dot_product_attention(q, kk, vv)


print(f"[A] TorchScript 编译通过, device={DEV}")

MAXT = 64
blk = StaticBlock(H)

# ---------- B. 零拷贝检查 ----------
buf = torch.zeros(B, MAXT, D, device=DEV)
sl = buf.narrow(1, 0, 10)
viewed = sl.view(B, 10, H, Dh)
print(f"[B] narrow 后 view 成功: {viewed.shape}, "
      f"共享 storage={viewed.data_ptr() == buf.data_ptr()}, "
      f"是否 contiguous={sl.is_contiguous()}")

# ---------- C. 数值等价 ----------
torch.manual_seed(0)
PREFILL, STEPS = 8, 20

k_ref = torch.randn(B, PREFILL, D, device=DEV)
v_ref = torch.randn(B, PREFILL, D, device=DEV)
new_k = [torch.randn(B, 1, D, device=DEV) for _ in range(STEPS)]
new_v = [torch.randn(B, 1, D, device=DEV) for _ in range(STEPS)]
qs = [torch.randn(B, 1, D, device=DEV) for _ in range(STEPS)]

# 基线: torch.cat 逐步增长（现状实现）
kc, vc = k_ref.clone(), v_ref.clone()
cat_out = []
for i in range(STEPS):
    kc = torch.cat([kc, new_k[i]], dim=1)
    vc = torch.cat([vc, new_v[i]], dim=1)
    L = kc.shape[1]
    q_ = qs[i].view(B, 1, H, Dh).transpose(1, 2)
    k_ = kc.view(B, L, H, Dh).transpose(1, 2)
    v_ = vc.view(B, L, H, Dh).transpose(1, 2)
    cat_out.append(F.scaled_dot_product_attention(q_, k_, v_))

# 方案1: (B,T,D) 静态 buffer
kb = torch.zeros(B, MAXT, D, device=DEV)
vb = torch.zeros(B, MAXT, D, device=DEV)
kb.narrow(1, 0, PREFILL).copy_(k_ref)
vb.narrow(1, 0, PREFILL).copy_(v_ref)
btd_out = [blk.step_btd(qs[i], new_k[i], new_v[i], kb, vb, PREFILL + i) for i in range(STEPS)]

# 方案2: (B,H,T,d) 静态 buffer
kb2 = torch.zeros(B, H, MAXT, Dh, device=DEV)
vb2 = torch.zeros(B, H, MAXT, Dh, device=DEV)
kb2.narrow(2, 0, PREFILL).copy_(k_ref.view(B, PREFILL, H, Dh).transpose(1, 2))
vb2.narrow(2, 0, PREFILL).copy_(v_ref.view(B, PREFILL, H, Dh).transpose(1, 2))
bhtd_out = []
for i in range(STEPS):
    q_ = qs[i].view(B, 1, H, Dh).transpose(1, 2)
    k_ = new_k[i].view(B, 1, H, Dh).transpose(1, 2)
    v_ = new_v[i].view(B, 1, H, Dh).transpose(1, 2)
    bhtd_out.append(blk.step_bhtd(q_, k_, v_, kb2, vb2, PREFILL + i))

d1 = max((a - b).abs().max().item() for a, b in zip(cat_out, btd_out))
d2 = max((a - b).abs().max().item() for a, b in zip(cat_out, bhtd_out))
print(f"[C] 最大偏差 vs torch.cat 基线:  (B,T,D)方案={d1:.3e}   (B,H,T,d)方案={d2:.3e}")
print("[C] 结论:", "两种布局均数值等价" if max(d1, d2) < 1e-4 else "!!! 有偏差，需排查")

# ---------- D. 驱动内存对照（MPS only）----------
if DEV == "mps":
    torch.mps.empty_cache()
    base = torch.mps.driver_allocated_memory()
    kc2, vc2 = torch.randn(B, 600, D, device=DEV), torch.randn(B, 600, D, device=DEV)
    for _ in range(600):
        kc2 = torch.cat([kc2, torch.randn(B, 1, D, device=DEV)], dim=1)
        vc2 = torch.cat([vc2, torch.randn(B, 1, D, device=DEV)], dim=1)
    cat_mem = torch.mps.driver_allocated_memory() - base
    del kc2, vc2
    torch.mps.empty_cache()
    base = torch.mps.driver_allocated_memory()
    sb_k = torch.zeros(B, 1300, D, device=DEV)
    sb_v = torch.zeros(B, 1300, D, device=DEV)
    for i in range(600):
        sb_k.narrow(1, 600 + i, 1).copy_(torch.randn(B, 1, D, device=DEV))
        sb_v.narrow(1, 600 + i, 1).copy_(torch.randn(B, 1, D, device=DEV))
    static_mem = torch.mps.driver_allocated_memory() - base
    print(f"[D] 600步单层驱动内存增量: torch.cat={cat_mem / 1e6:.1f}MB  静态={static_mem / 1e6:.1f}MB "
          f"({cat_mem / max(static_mem, 1):.1f}x)")
