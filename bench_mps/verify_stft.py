"""STFT：验证 torch.stft 的 return_complex=False → True+view_as_real 改动。

⚠️ 这个脚本在 2026-09-03 的审计中被修正过一次，教训值得留着看：
原版只预热1次、且固定「先跑 False 再跑 True」，测出 True 快 2.5x（0.333→0.133ms），
这个数字一度被当作结论。审计时把两者顺序对调，发现**先跑的那个
永远是 0.33ms、后跑的永远是 0.135ms**——2.5x 完全是预热假象，两者速度其实一样。

所以：MPS 上做微基准，必须 (1) 充分预热（一次不够）(2) 交替顺序跑多轮取中位数。
本脚本现在两条都做了。
"""

import statistics
import time

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# v2ProPlus 里 spectrogram_torch 的真实参数
N_FFT, HOP, WIN, SR = 2048, 640, 2048, 32000

y = torch.randn(1, SR * 8, device=DEV).clamp(-1, 1)
window = torch.hann_window(WIN, device=DEV)


def stft(return_complex: bool):
    spec = torch.stft(
        y, N_FFT, hop_length=HOP, win_length=WIN, window=window,
        center=False, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=return_complex,
    )
    return torch.view_as_real(spec) if return_complex else spec


# ---------- 1. 数值等价 ----------
diff = (stft(False) - stft(True)).abs().max().item()
print(f"[数值] max|old - new| = {diff:.3e}   {'逐位相同' if diff == 0.0 else '!!! 有偏差'}")

# ---------- 2. 性能（充分预热 + 交替顺序 + 多轮中位数）----------
for _ in range(10):  # 预热：1次不够，MPS 首轮有 kernel 编译/惰性初始化开销
    stft(False)
    stft(True)
if DEV == "mps":
    torch.mps.synchronize()

N, REPS = 50, 5
results = {False: [], True: []}
for rep in range(REPS):
    order = [False, True] if rep % 2 == 0 else [True, False]  # 交替，抵消先后偏差
    for rc in order:
        if DEV == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            stft(rc)
        if DEV == "mps":
            torch.mps.synchronize()
        results[rc].append((time.perf_counter() - t0) / N * 1000)

for rc in (False, True):
    runs = " ".join(f"{v:.3f}" for v in results[rc])
    print(f"[性能] return_complex={str(rc):5s} 中位数 {statistics.median(results[rc]):.3f} ms/call   (各轮: {runs})")

print(
    "\n结论：两者性能相同。保留这个改动的理由是 return_complex=False 已废弃，"
    "PyTorch 警告原文说未来版本会直接报错——是为了避免将来升级 torch 时炸掉，不是为了性能。"
)
