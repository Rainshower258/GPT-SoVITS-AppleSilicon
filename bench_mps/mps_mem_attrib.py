"""按方案分别归因 MPS driver 内存（真正对应"内存涨到 14GB"的那个指标）。
每个方案在子进程里独立跑，避免相互污染。"""

import subprocess
import sys
import textwrap

COMMON = textwrap.dedent("""
    import time, torch, torch.nn.functional as F
    N_LAYER, HIDDEN, N_HEAD = 24, 512, 16
    PREFIX, STEPS = 200, 600
    DEV = torch.device("mps")
    def cur():  return torch.mps.current_allocated_memory()/1024**2
    def drv():  return torch.mps.driver_allocated_memory()/1024**2
    _ = torch.randn(512,512,device=DEV) @ torch.randn(512,512,device=DEV)
    torch.mps.synchronize()
    base_drv = drv()
""")

CAT = COMMON + textwrap.dedent("""
    k=[torch.randn(1,PREFIX,HIDDEN,device=DEV) for _ in range(N_LAYER)]
    v=[torch.randn(1,PREFIX,HIDDEN,device=DEV) for _ in range(N_LAYER)]
    eos=torch.tensor(1024,device=DEV); peak_d=drv()
    torch.mps.synchronize(); t0=time.perf_counter()
    for s in range(STEPS):
        x=torch.randn(1,1,HIDDEN,device=DEV)
        for i in range(N_LAYER):
            kk_=torch.randn(1,1,HIDDEN,device=DEV); vv_=torch.randn(1,1,HIDDEN,device=DEV)
            k[i]=torch.cat([k[i],kk_],dim=1); v[i]=torch.cat([v[i],vv_],dim=1)
            L=k[i].shape[1]
            q=x.view(1,1,N_HEAD,-1).transpose(1,2)
            a=F.scaled_dot_product_attention(q, k[i].view(1,L,N_HEAD,-1).transpose(1,2),
                                                v[i].view(1,L,N_HEAD,-1).transpose(1,2))
            x=a.transpose(1,2).reshape(1,1,HIDDEN)
        smp=torch.randint(0,1024,(1,1),device=DEV)
        if SYNC: _=bool((smp[:,0]==eos).any())
        if s%25==0: peak_d=max(peak_d,drv())
    torch.mps.synchronize(); dt=time.perf_counter()-t0; peak_d=max(peak_d,drv())
    print(f"{dt:.2f} {STEPS/dt:.1f} {cur():.1f} {peak_d:.1f} {base_drv:.1f}")
""")

STATIC = COMMON + textwrap.dedent("""
    ML=PREFIX+STEPS
    k=torch.zeros(N_LAYER,1,ML,HIDDEN,device=DEV); v=torch.zeros(N_LAYER,1,ML,HIDDEN,device=DEV)
    k[:,:,:PREFIX]=torch.randn(N_LAYER,1,PREFIX,HIDDEN,device=DEV)
    v[:,:,:PREFIX]=torch.randn(N_LAYER,1,PREFIX,HIDDEN,device=DEV)
    eos=torch.tensor(1024,device=DEV); peak_d=drv()
    torch.mps.synchronize(); t0=time.perf_counter()
    for s in range(STEPS):
        p=PREFIX+s; x=torch.randn(1,1,HIDDEN,device=DEV)
        for i in range(N_LAYER):
            k[i,:,p:p+1]=torch.randn(1,1,HIDDEN,device=DEV)
            v[i,:,p:p+1]=torch.randn(1,1,HIDDEN,device=DEV)
            L=p+1
            q=x.view(1,1,N_HEAD,-1).transpose(1,2)
            a=F.scaled_dot_product_attention(q, k[i,:,:L].view(1,L,N_HEAD,-1).transpose(1,2),
                                                v[i,:,:L].view(1,L,N_HEAD,-1).transpose(1,2))
            x=a.transpose(1,2).reshape(1,1,HIDDEN)
        smp=torch.randint(0,1024,(1,1),device=DEV)
        if SYNC: _=bool((smp[:,0]==eos).any())
        if s%25==0: peak_d=max(peak_d,drv())
    torch.mps.synchronize(); dt=time.perf_counter()-t0; peak_d=max(peak_d,drv())
    print(f"{dt:.2f} {STEPS/dt:.1f} {cur():.1f} {peak_d:.1f} {base_drv:.1f}")
""")

CASES = [
    ("A  cat cache + 每步同步 (当前代码)", CAT, True),
    ("A' cat cache + 无同步", CAT, False),
    ("B  静态 cache + 每步同步", STATIC, True),
    ("B' 静态 cache + 无同步 (最优)", STATIC, False),
]

print(f"{'方案':<34} {'耗时':>7} {'it/s':>8} {'张量内存':>10} {'驱动峰值':>11}")
print("-" * 76)
for name, src, sync in CASES:
    out = subprocess.run(
        [sys.executable, "-c", f"SYNC={sync}\n" + src], capture_output=True, text=True
    )
    if out.returncode != 0:
        print(f"{name:<34}  失败: {out.stderr.strip()[-200:]}")
        continue
    dt, its, cur, drv, base = out.stdout.split()
    print(f"{name:<34} {dt:>6}s {its:>8} {cur:>8} MB {drv:>8} MB")
