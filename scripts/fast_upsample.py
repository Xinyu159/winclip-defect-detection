"""双线性上采样的向量化实现 + 与参考实现的强制一致性验证。

为什么需要:`eval_ov.upsample_bilinear_np` 是逐像素 Python 双重循环
(240×240 = 57600 次标量运算/图)。cascade_v2 的网格扫描要对
(1258 缺陷 + 467 良品) × 24 组配置 反复上采样 → 72000 次 × ~75ms
≈ 1.5 小时,纯属浪费。

这是**纯性能优化,不改语义**:调用方 `verify_fast_upsample()` 必须在
首次使用前跑一次,与参考实现逐位比对;不一致就抛异常而不是静默用快版本。
AUROC 对排序敏感,上采样权重错一点点就可能换序,所以这条闸门不能省。

数学与参考实现逐条对齐(align_corners=False 的 F.interpolate 镜像):
    y = (dy+0.5)*h/dst_h - 0.5 ; y0 = floor(y) ; λy = y - y0(未裁剪)
    索引裁剪到 [0,h-1];y1 = min(y0c+1, h-1)
注意 λ 用**未裁剪**的 y0 计算,只有索引裁剪 —— 参考实现即如此。
"""
from __future__ import annotations

import numpy as np


def upsample_bilinear_fast(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """(h,w) → (dst_h,dst_w) 双线性,语义同 eval_ov.upsample_bilinear_np。"""
    h, w = src.shape
    ys = (np.arange(dst_h, dtype=np.float64) + 0.5) * (h / dst_h) - 0.5
    xs = (np.arange(dst_w, dtype=np.float64) + 0.5) * (w / dst_w) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    ly = ys - y0                                    # 未裁剪
    lx = xs - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y0c + 1, 0, h - 1)
    x1c = np.clip(x0c + 1, 0, w - 1)

    s = src.astype(np.float64, copy=False)
    row0, row1 = s[y0c], s[y1c]                     # (dst_h, w)
    lxf, rxf = (1.0 - lx)[None, :], lx[None, :]
    top = row0[:, x0c] * lxf + row0[:, x1c] * rxf   # (dst_h, dst_w)
    bot = row1[:, x0c] * lxf + row1[:, x1c] * rxf
    lyf = ly[:, None]
    return (1.0 - lyf) * top + lyf * bot


def verify_fast_upsample(shape=(15, 15), dst=(240, 240), n: int = 3,
                         seed: int = 0, verbose: bool = True) -> float:
    """与参考实现逐位比对,返回最大绝对差。不一致由调用方决定如何处理。"""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.eval_ov import upsample_bilinear_np

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        src = rng.random(shape)
        ref = upsample_bilinear_np(src, *dst)
        fast = upsample_bilinear_fast(src, *dst)
        worst = max(worst, float(np.abs(ref - fast).max()))
    if verbose:
        print(f"[upsample] 向量化 vs 参考 max|Δ| = {worst:.2e}", flush=True)
    return worst
