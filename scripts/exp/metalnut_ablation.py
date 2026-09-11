"""metal_nut 反向的**机制消融**:到底是哪一项把像素图弄坏的?

## 现象(15 类全量实测)

metal_nut 是唯一"窗口塔让它变差"的类,但**只在判定口径变差**:

| 口径 | L1 | full | Δ |
|---|---|---|---|
| 定位 pix(few_map only) | 90.2 | 92.2 | **+2.0** |
| 判定 pix(m_all)         | 74.1 | 69.9 | **−4.2** |

定位明明变好了,判定却变差 → 损害**不是窗口塔引入的**,而是引入在
`m_all` 的**组装方式**上。`m_all` 与 `few_map` 的差别只有一处:
  `m_all = 调和(cls_prob, 各窗口 zero 分) + few_map`
  `few_map` 本身 = 各尺度 few 分的算术平均

本地报告的原推测是"大范围形变 → 调和平均被窗内小值支配"。
但该推测在**跨类**层面已被证伪(15 类 Spearman(GT面积, Δ判定img)=+0.03,
p=0.93;screw 面积最小反而掉最多)。所以必须另找机制。

## 本脚本的假设与消融

**假设 H:坏在 `cls_prob` 那一项**。调和平均被**小值**支配,而
`cls_prob` 在缺陷图上接近 1、`1/cls_prob` 很小 → 它对 inv 的贡献可忽略,
m_all 于是被**窗口值**支配。大范围形变类的窗口 CLS 特征把缺陷+背景
平均掉 → 窗口分偏低 → 缺陷 patch 被拉到接近背景。

**消融 4 个变体**(全部只用缓存特征,不碰 GPU):
  A 原式     : `调和(cls_prob, 窗口) + few_map`
  B 去掉cls  : `调和(窗口) + few_map`          —— 若 B≫A 则 H 成立
  C cls加性  : `cls_prob + 调和(窗口) + few_map` —— 与 B 对照看 cls 是灌水还是灌毒
  D 只3×3/只2×2 分开看,定位是哪一尺度带来的

用法:
    python scripts/exp/metalnut_ablation.py --classes metal_nut,screw,tile
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/autodl-tmp/winclip")
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P   # noqa: E402
from scripts.fast_upsample import upsample_bilinear_fast      # noqa: E402

CD = Path("/root/autodl-tmp/feat_cache_gpu")
CG = Path("/root/autodl-tmp/feat_cache_good")
TD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/text_protos")
few = P._few_token_score


def prep(f: Path, gal, pos, neg, temp):
    z = np.load(f)
    return dict(cls_prob=float(P._prob(z["full"][:1], pos, neg, temp)[0]),
                base=few(z["full"][1:], gal["patch"]),
                f3=few(z["w3"], gal["large"]), f5=few(z["w5"], gal["mid"]),
                z3=P._prob(z["w3"], pos, neg, temp),
                z5=P._prob(z["w5"], pos, neg, temp),
                gt=z["gt"] if "gt" in z.files else None)


def maps(p, idx3, idx2, use_cls="harmonic", scales=("w3", "w5")):
    """返回 (few_map, m_all_variant)。use_cls ∈ {harmonic, none, additive}。"""
    num = p["base"].copy()
    den = np.ones(N_PATCH, np.float32)
    for key, idx in (("w3", idx3), ("w5", idx2)):
        if key not in scales:
            continue
        m, c = P._scatter_harmonic(p["f" + key[-1]], idx)
        pr = c > 0
        num[pr] += m[pr]
        den[pr] += 1.0
    few_map = num / den

    if not scales:
        win = np.zeros(N_PATCH, np.float32)
    else:
        if use_cls == "none":
            inv = np.zeros(N_PATCH, np.float32)
            n = np.zeros(N_PATCH, np.float32)
        else:
            inv = np.full(N_PATCH, 1.0 / p["cls_prob"], np.float32)
            n = np.ones(N_PATCH, np.float32)
        for key, idx in (("w3", idx3), ("w5", idx2)):
            if key not in scales:
                continue
            m, c = P._scatter_harmonic(p["z" + key[-1]], idx)
            pr = c > 0
            inv[pr] += 1.0 / m[pr]
            n[pr] += 1.0
        win = (n / inv).astype(np.float32)
        if use_cls == "none":
            # 未被任何窗口覆盖的 patch:inv=0 → 用 0 表示"无窗口证据"
            win = np.where(n > 0, win, 0.0).astype(np.float32)
    if use_cls == "additive":
        win = win + p["cls_prob"]
    return few_map, win + few_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="metal_nut,screw,tile")
    ap.add_argument("--out", default="/tmp/metalnut_ablation.json")
    args = ap.parse_args()

    idx3 = np.load("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/win_idx_k3.npy")
    idx2 = np.load("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/win_idx_k2.npy")
    out = {}
    for cls in args.classes.split(","):
        d = np.load(TD / f"{cls}.npz")
        pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
        gal = dict(np.load(CD / cls / "gallery.npz"))
        recs = []
        for f in sorted((CD / cls).glob("[0-9]*.npz")):
            p = prep(f, gal, pos, neg, temp)
            if p["gt"] is not None and p["gt"].any():
                recs.append(p)
        if not recs:
            continue
        gt = np.concatenate([r["gt"].ravel() for r in recs])

        variants = {
            "A_原式(调和cls+窗)": ("harmonic", ("w3", "w5")),
            "B_去cls(只窗)": ("none", ("w3", "w5")),
            "C_cls加性": ("additive", ("w3", "w5")),
            "D_只3x3": ("harmonic", ("w3",)),
            "E_只2x2": ("harmonic", ("w5",)),
        }
        row = {}
        # 定位基线:few_map(与 A 同尺度,但不含 zero 分支)
        fm0, m0 = maps(recs[0], idx3, idx2)
        loc = np.concatenate([upsample_bilinear_fast(
            maps(r, idx3, idx2)[0].reshape(GRID, GRID), 240, 240).ravel()
            for r in recs])
        row["loc_pix(few only)"] = round(
            float(roc_auc_score(gt, loc)) * 100, 1)
        # L1(无窗口)的判定图 = cls_prob(常数) + few_map
        l1 = np.concatenate([upsample_bilinear_fast(
            (np.full(N_PATCH, r["cls_prob"], np.float32) +
             maps(r, idx3, idx2)[0]).reshape(GRID, GRID), 240, 240).ravel()
            for r in recs])
        row["L1_判定pix(cls+few)"] = round(
            float(roc_auc_score(gt, l1)) * 100, 1)

        for name, (uc, sc) in variants.items():
            mm = np.concatenate([upsample_bilinear_fast(
                maps(r, idx3, idx2, uc, sc)[1].reshape(GRID, GRID),
                240, 240).ravel() for r in recs])
            row[name] = round(float(roc_auc_score(gt, mm)) * 100, 1)
        out[cls] = row
        print(f"\n[{cls}]  n_defect={len(recs)}")
        for k, v in row.items():
            print(f"   {k:24s} {v:6.1f}")

    print("\n=== 结论 ===")
    for cls, r in out.items():
        a = r["A_原式(调和cls+窗)"]
        b = r["B_去cls(只窗)"]
        c = r["C_cls加性"]
        print(f"{cls:11s} A={a:5.1f}  B={b:5.1f}({b-a:+.1f})  "
              f"C={c:5.1f}({c-a:+.1f})  定位={r['loc_pix(few only)']:5.1f}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
