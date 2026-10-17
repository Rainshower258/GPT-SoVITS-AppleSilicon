"""前置探测：fp16/bf16 在 T2S decode 的真实形状上，MPS 到底能快多少。
形状取自 s1longer-v2.yaml: n_layer=24, hidden=512, head=16, head_dim=32。
按 mps-benchmarking-discipline: 充分预热 + 交替顺序 + 多轮中位数。"""
import statistics, time, torch
import torch.nn.functional as F

DEV = "mps"
L, D, H, Dh = 24, 512, 16, 32
KV = 1400          # 长文本解码后期的典型 kv 长度；改成 250 可看短句场景
B = 1

def make(dtype):
    g = torch.Generator(device="cpu").manual_seed(0)
    w_qkv = [torch.randn(3*D, D, generator=g).to(DEV, dtype) for _ in range(L)]
    w_out = [torch.randn(D, D, generator=g).to(DEV, dtype) for _ in range(L)]
    w1 = [torch.randn(4*D, D, generator=g).to(DEV, dtype) for _ in range(L)]
    w2 = [torch.randn(D, 4*D, generator=g).to(DEV, dtype) for _ in range(L)]
    kc = [torch.zeros(B, KV+8, D, device=DEV, dtype=dtype) for _ in range(L)]
    vc = [torch.zeros(B, KV+8, D, device=DEV, dtype=dtype) for _ in range(L)]
    x  = torch.randn(B, 1, D, generator=g).to(DEV, dtype)
    return w_qkv, w_out, w1, w2, kc, vc, x

def step(pack, kv_len):
    w_qkv, w_out, w1, w2, kc, vc, x = pack
    for i in range(L):
        q, k, v = F.linear(x, w_qkv[i]).chunk(3, dim=-1)
        kc[i].narrow(1, kv_len, 1).copy_(k)
        vc[i].narrow(1, kv_len, 1).copy_(v)
        n = kv_len + 1
        qq = q.view(B,1,H,Dh).transpose(1,2)
        kk = kc[i].narrow(1,0,n).view(B,n,H,Dh).transpose(1,2)
        vv = vc[i].narrow(1,0,n).view(B,n,H,Dh).transpose(1,2)
        a = F.scaled_dot_product_attention(qq, kk, vv)
        a = a.transpose(1,2).reshape(B,1,D)
        x = x + F.linear(a, w_out[i])
        x = F.layer_norm(x, [D])
        x = x + F.linear(F.relu(F.linear(x, w1[i])), w2[i])
        x = F.layer_norm(x, [D])
    return x

dtypes = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]
packs = {}
for name, dt in dtypes:
    try:
        packs[name] = make(dt)
        step(packs[name], KV-1)      # 冒烟
        print(f"{name}: 支持")
    except Exception as e:
        print(f"{name}: 不支持 -> {type(e).__name__}: {e}")

names = [n for n in packs]
for _ in range(10):                   # 充分预热
    for n in names: step(packs[n], KV-1)
torch.mps.synchronize()

N, REPS = 30, 5
res = {n: [] for n in names}
for rep in range(REPS):
    order = names if rep % 2 == 0 else names[::-1]   # 交替顺序
    for n in order:
        torch.mps.synchronize(); t = time.perf_counter()
        for _ in range(N): step(packs[n], KV-1)
        torch.mps.synchronize()
        res[n].append((time.perf_counter()-t)/N*1000)

print(f"\n=== 24层 decode 单步耗时 (kv_len={KV}) ===")
base = statistics.median(res["fp32"])
for n in names:
    m = statistics.median(res[n])
    print(f"{n}: {m:.3f} ms/step   相对fp32 {base/m:.2f}x   (各轮 {' '.join(f'{v:.3f}' for v in res[n])})")
