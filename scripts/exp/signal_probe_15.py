# 逐个中间信号的独立 AUROC —— 全 15 类,256 全量缺陷图。
# 回答 T01 修订版 §1.5 的第 3 问:zero-shot 口径下 full 反而更差,原因是什么。
#
# 动机:级联的选窗依据曾用 `_prob(mosaic patch, 文本原型)`,实测与 GT 反相关。
# 在修任何东西之前先把每个零件单独测一遍,否则会拿一个坏零件去解释另一个坏现象。
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/autodl-tmp/winclip")
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P   # noqa: E402
# 向量化版本;与 upsample_bilinear_np 逐位一致(max|Δ|=0,见 fast_upsample)。
# 参考实现是纯 Python 双重循环,1258 张图 × 5 个信号会跑成小时级。
from scripts.fast_upsample import upsample_bilinear_fast    # noqa: E402

CD = Path("/root/autodl-tmp/feat_cache_gpu")
TD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/text_protos")
DD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn")
idx3 = np.load(DD / "win_idx_k3.npy")
idx2 = np.load(DD / "win_idx_k2.npy")
harm, few = P._scatter_harmonic, P._few_token_score
CLASSES = ["tile", "carpet", "bottle", "metal_nut", "screw",
           "cable", "capsule", "grid", "hazelnut", "leather",
           "pill", "toothbrush", "transistor", "wood", "zipper"]


def A(ps, pg):
    return roc_auc_score(np.concatenate(pg), np.concatenate(ps)) * 100


hdr = ("类", "mosaicText", "mosaicGal", "winZero", "winFew", "cv")
print(f"{hdr[0]:10s} {hdr[1]:>11s} {hdr[2]:>11s} {hdr[3]:>10s} "
      f"{hdr[4]:>10s} {hdr[5]:>10s}", flush=True)
means = {k: [] for k in hdr[1:]}

for cls in CLASSES:
    d = np.load(TD / f"{cls}.npz")
    pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
    gal = dict(np.load(CD / cls / "gallery.npz"))
    acc = {k: [[], []] for k in hdr[1:]}
    for f in sorted((CD / cls).glob("[0-9]*.npz")):
        z = np.load(f)
        gt = z["gt"]
        if not gt.any():
            continue
        full, w3, w5, susp = z["full"], z["w3"], z["w5"], z["susp"]

        # ① 整图 patch 令牌 × 文本原型(被证伪的那条路)
        mapt = P._prob(full[1:], pos, neg, temp)
        # ② 整图 patch 令牌 × gallery(同空间,地基)
        mapg = few(full[1:], gal["patch"])
        # ③ 全量窗 × 文本原型,调和平均组装(zero-shot 的窗口路径)
        m3, c3 = harm(P._prob(w3, pos, neg, temp), idx3)
        m5, c5 = harm(P._prob(w5, pos, neg, temp), idx2)
        inv = np.full(N_PATCH, 1e12, np.float32)
        nt = np.zeros(N_PATCH, np.float32)
        for m, c in ((m3, c3), (m5, c5)):
            pr = c > 0
            inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
            nt[pr] += 1.0
        wz = np.where(nt > 0, nt / inv, 0.0)
        # ④ 全量窗 × gallery(research few-shot 的窗口路径)
        f3, c3f = harm(few(w3, gal["large"]), idx3)
        f5, c5f = harm(few(w5, gal["mid"]), idx2)
        num = few(full[1:], gal["patch"]).copy()
        den = np.ones(N_PATCH, np.float32)
        for m, c in ((f3, c3f), (f5, c5f)):
            pr = c > 0
            num[pr] += m[pr]
            den[pr] += 1.0
        wf = num / den

        for k, arr in (("mosaicText", mapt), ("mosaicGal", mapg),
                       ("winZero", wz), ("winFew", wf),
                       ("cv", susp.ravel())):
            up = upsample_bilinear_fast(arr.reshape(GRID, GRID), 240, 240)
            acc[k][0].append(up.flatten())
            acc[k][1].append(gt.flatten())
    vals = {k: A(*acc[k]) for k in hdr[1:]}
    for k in hdr[1:]:
        means[k].append(vals[k])
    print(f"{cls:10s} {vals['mosaicText']:11.1f} {vals['mosaicGal']:11.1f} "
          f"{vals['winZero']:10.1f} {vals['winFew']:10.1f} "
          f"{vals['cv']:10.1f}", flush=True)

print()
print(f"{'15类均值':10s} " + " ".join(
    f"{np.mean(means[k]):11.1f}" if i == 0 else f"{np.mean(means[k]):10.1f}"
    for i, k in enumerate(hdr[1:])))
print("\n(winFew 全量 = research 路径 few-shot 用的信号;"
      " mosaicGal = few-shot 里 patch 尺度那一路)")
