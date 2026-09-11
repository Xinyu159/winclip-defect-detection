"""三档复核 + GT 面积机制分析(任务书 T01 §1.5 首要目的)。

## 为什么另跑一遍,不在 cascade_prod_merge.py 上加档位

`cascade_prod_merge.py` 的预算 B 是 **3×3 窗口数**,2×2 窗口数按
`n5 = int(B * 196/169)` 线性配比。于是 B=169 时 n5 已经饱和到 196 ——
**全量**。那份脚本的结构上就表达不出"只开 3×3、不开 2×2"这一档。

任务书 §1.5 的三档按 token 数可精确反推(这是本次复核的锚点):
  L1       = 226                    = patcher(1) + 整图塔(225),**零窗口**
  L1+3×3   = 226 + 169×10 = 1916    (3×3 窗 = 9 patch + 1 cls = 10 token)
  full     = 1916 + 196×5 = 2896    (2×2 窗 = 4 patch + 1 cls =  5 token)
→ 三档 = {不开窗, 只开 w3, w3+w5 全开}。本脚本按此定义显式分档。

## 顺带修掉上一份的一个口径 bug

`cascade_prod_merge.py` 里 B=0 分支写成 `s = cls_prob`,而产线
`anomaly_maps` 在 `use_few=True` 时**恒**为 `(cls_prob + few_map.max())/2`,
B=0 时 few_map 退化为 patch 分支、**不是**空。所以那份的"纯地基"列
实际测的是**纯 cls_prob 的 AUROC**(= zero-shot 基线),不是产线在无窗口
时的判定分数。本脚本两者都算并分列,避免再把基线当产线分数。

## 回答任务书两个问题

Q1 其余 10 类(img 偏低的 cable/capsule/pill 等)是否也呈现"窗口塔救不了
   图像级" → 逐类列出 定位img / 判定img 在三档上的走向,并用
   `drag = 判定img − 定位img` 量化 **cls_prob 项对 few 信号的拖累**
   (`img_score=(cls_prob+max)/2` 是两者的算术平均,cls_prob 判别力弱就会
   把 few 的强信号摊薄 —— 这正是"救不了"的机制候选)

Q2 metal_nut 反向的机制 → 统计缺陷 **GT 面积占比**,做两级相关:
   跨类:每类平均 GT 面积 vs 该类 Δimg / Δpix
   类内:每张图的 GT 面积 vs 该图自身的分数增益
   推测是"大范围形变 → 调和平均(1/m 对小值敏感)被窗内小值支配" →
   若成立应看到 **负相关**(面积越大,增益越小)

## 过杀/漏检口径(任务书坑三)

`过杀率@Pxx` 是同义反复(阈值本取自良品 Pxx 分位)。本脚本只输出
**(过杀固定, 漏检) 配对**:阈值取良品某分位 → 报该阈值下的实际过杀率
与漏检率。并显式报每类良品数 N(过杀最小粒度 = 1/N)。

用法:
    python scripts/exp/cascade_tier_gtarea.py --classes all --out /tmp/tier_gt.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/autodl-tmp/winclip")
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P   # noqa: E402
from scripts.fast_upsample import upsample_bilinear_fast      # noqa: E402

CD = Path("/root/autodl-tmp/feat_cache_gpu")
CG = Path("/root/autodl-tmp/feat_cache_good")
TD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/text_protos")
DD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn")
few = P._few_token_score
CLASSES = ["tile", "carpet", "bottle", "metal_nut", "screw", "cable", "capsule",
           "grid", "hazelnut", "leather", "pill", "toothbrush", "transistor",
           "wood", "zipper"]
# 本地那张 5 类表的类(用于对照本地结论)
LOCAL5 = ["cable", "capsule", "pill", "screw", "metal_nut"]

# 三档:只开哪些尺度的**全部**窗口
TIERS = {
    "L1": (),            # 零窗口          226 token / 1.0x
    "L1+3x3": ("w3",),   # 只开 3×3        1916 token / 8.5x
    "full": ("w3", "w5"),  # 两尺度全开    2896 token / 12.8x
}
TIER_TOKENS = {"L1": 226, "L1+3x3": 1916, "full": 2896}


def precompute(f: Path, gal: dict, pos, neg, temp):
    """一图一次:算出与档位无关的全部中间量。"""
    z = np.load(f)
    full = z["full"]
    return dict(
        base=few(full[1:], gal["patch"]),          # patch 分支 few (225,)
        f3=few(z["w3"], gal["large"]),             # 3×3 few (169,)
        f5=few(z["w5"], gal["mid"]),               # 2×2 few (196,)
        z3=P._prob(z["w3"], pos, neg, temp),       # 3×3 zero (169,)
        z5=P._prob(z["w5"], pos, neg, temp),       # 2×2 zero (196,)
        cls_prob=float(P._prob(full[:1], pos, neg, temp)[0]),
        gt=z["gt"] if "gt" in z.files else None,
    )


def few_map_of(pre, tiers, idx3, idx2):
    """定位口径:只留 few_map(算术平均,与产线 few 分支同规则)。"""
    num = pre["base"].copy()
    den = np.ones(N_PATCH, np.float32)
    if "w3" in tiers:
        m, c = P._scatter_harmonic(pre["f3"], idx3)
        pr = c > 0
        num[pr] += m[pr]
        den[pr] += 1.0
    if "w5" in tiers:
        m, c = P._scatter_harmonic(pre["f5"], idx2)
        pr = c > 0
        num[pr] += m[pr]
        den[pr] += 1.0
    return num / den


def judge_map_of(pre, tiers, idx3, idx2):
    """判定口径像素图:`m_all = 调和(cls_prob, 窗口zero) + few_map`。"""
    fm = few_map_of(pre, tiers, idx3, idx2)
    if not tiers:
        return np.full(N_PATCH, pre["cls_prob"], np.float32) + fm
    inv = np.full(N_PATCH, 1.0 / pre["cls_prob"], np.float32)
    n = np.ones(N_PATCH, np.float32)
    for key, idx in (("w3", idx3), ("w5", idx2)):
        if key not in tiers:
            continue
        m, c = P._scatter_harmonic(pre[f"z{3 if key == 'w3' else 5}"], idx)
        pr = c > 0
        inv[pr] += 1.0 / m[pr]
        n[pr] += 1.0
    return (n / inv).astype(np.float32) + fm


def ups(m):
    return upsample_bilinear_fast(m.reshape(GRID, GRID), 240, 240).ravel()


def auc(pos, neg):
    """pos=正类(缺陷)分数, neg=良品分数。"""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    return float(roc_auc_score(y, s)) * 100


def pair_at(neg, pos, q):
    """(过杀固定, 漏检) 配对:阈值取良品 q 分位。返回 (过杀%, 漏检%)。"""
    if len(neg) == 0:
        return float("nan"), float("nan")
    thr = float(np.percentile(neg, q))
    return (float((neg > thr).mean()) * 100,
            float((pos <= thr).mean()) * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tier_gt.json")
    ap.add_argument("--classes", default="all")
    args = ap.parse_args()
    classes = CLASSES if args.classes == "all" else args.classes.split(",")
    idx3 = np.load(DD / "win_idx_k3.npy")
    idx2 = np.load(DD / "win_idx_k2.npy")

    out = {"tiers": {k: {"tokens": TIER_TOKENS[k], "scales": list(v)}
                     for k, v in TIERS.items()},
           "results": {}, "per_image_area": {}}
    t0 = time.time()
    for cls in classes:
        d = np.load(TD / f"{cls}.npz")
        pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
        gal = dict(np.load(CD / cls / "gallery.npz"))

        good, bad = [], []
        for f in sorted((CG / cls).glob("[0-9]*.npz")):
            good.append(precompute(f, gal, pos, neg, temp))
        for f in sorted((CD / cls).glob("[0-9]*.npz")):
            p = precompute(f, gal, pos, neg, temp)
            if p["gt"] is not None and p["gt"].any():
                bad.append(p)

        n_good, n_bad = len(good), len(bad)
        res = {"n_good": n_good, "n_bad": n_bad,
               # 过杀最小粒度:1/N(任务书"采样量学")
               "overkill_grain_pct": round(100.0 / n_good, 1) if n_good else None,
               "tiers": {}}
        for tname, tiers in TIERS.items():
            loc_g, loc_b, jud_g, jud_b = [], [], [], []
            cls_g, cls_b = [], []          # 纯 cls_prob(与档位无关, 作参照)
            pixj, pixl, pixg = [], [], []  # 判定像素图 / 定位像素图 / GT
            for tag, recs in (("g", good), ("b", bad)):
                for p in recs:
                    fm = few_map_of(p, tiers, idx3, idx2)
                    # 定位口径:纯 few_map.max(),不含全局标量
                    (loc_g if tag == "g" else loc_b).append(float(fm.max()))
                    # 判定口径:产线公式,**B=0 时 few_map 退化为 patch 分支,不置零**
                    (jud_g if tag == "g" else jud_b).append(
                        (p["cls_prob"] + float(fm.max())) / 2.0)
                    (cls_g if tag == "g" else cls_b).append(p["cls_prob"])
                    if tag == "b":
                        pixj.append(ups(judge_map_of(p, tiers, idx3, idx2)))
                        pixl.append(ups(fm))
                        pixg.append(p["gt"].ravel())
            loc_g, loc_b = np.array(loc_g), np.array(loc_b)
            jud_g, jud_b = np.array(jud_g), np.array(jud_b)
            gt = np.concatenate(pixg)
            ok0, esc0 = pair_at(jud_g, jud_b, 0.0)       # 阈值=max(良品) → 过杀=0
            ok95, esc95 = pair_at(jud_g, jud_b, 95.0)    # 阈值=P95
            ok99, esc99 = pair_at(jud_g, jud_b, 99.0)    # 阈值=P99
            res["tiers"][tname] = {
                "loc_img": round(auc(loc_b, loc_g), 1),
                "jud_img": round(auc(jud_b, jud_g), 1),
                "cls_img": round(auc(np.array(cls_b), np.array(cls_g)), 1),
                # 定位 pix = 纯 few_map(缺陷"在哪里");判定 pix = m_all(产线用的图)
                "loc_pix": round(float(roc_auc_score(
                    gt, np.concatenate(pixl))) * 100, 1),
                "jud_pix": round(float(roc_auc_score(
                    gt, np.concatenate(pixj))) * 100, 1),
                # (过杀固定, 漏检) 配对 —— 不报"过杀@Pxx"这种同义反复
                "pair_overkill0_escape": round(esc0, 1),
                "pair_p95_overkill": round(ok95, 1),
                "pair_p95_escape": round(esc95, 1),
                "pair_p99_overkill": round(ok99, 1),
                "pair_p99_escape": round(esc99, 1),
            }
        # 逐图 GT 面积占比 + 该图在 L1→full 上的增益(类内相关用)
        areas, gains = [], []
        for p in bad:
            fm1 = few_map_of(p, (), idx3, idx2)
            fmf = few_map_of(p, ("w3", "w5"), idx3, idx2)
            area = float(p["gt"].mean())                 # 面积占比(0~1)
            areas.append(area)
            gains.append((float(fmf.max()) - float(fm1.max())) / 2.0)  # Δ判定分
        res["gt_area_mean"] = round(float(np.mean(areas)), 4)
        if len(areas) > 2 and np.std(gains) > 0 and np.std(areas) > 0:
            rho, pv = spearmanr(areas, gains)
            res["within_class_rho_area_vs_gain"] = round(float(rho), 3)
            res["within_class_rho_p"] = round(float(pv), 4)
        else:
            res["within_class_rho_area_vs_gain"] = None
            res["within_class_rho_p"] = None
        out["per_image_area"][cls] = {"area": areas, "gain": gains}
        out["results"][cls] = res
        print(f"[done] {cls:12s} 良品 {n_good:3d} 缺陷 {n_bad:3d} "
              f"GT面积 {res['gt_area_mean']:.3f} "
              f"类内rho {res['within_class_rho_area_vs_gain']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))

    # ---------------- 汇总 ----------------
    print(f"\n三档 × 两口径(15类均值)")
    hdr = f"{'档位':8s} {'token':>6s} {'定位img':>8s} {'判定img':>8s} "
    hdr += f"{'cls_img':>8s} {'定位pix':>8s} {'判定pix':>8s}"
    print(hdr)
    for t in TIERS:
        rows = [r["tiers"][t] for r in out["results"].values()]
        m = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        print(f"{t:8s} {TIER_TOKENS[t]:6d} {m('loc_img'):8.1f} "
              f"{m('jud_img'):8.1f} {m('cls_img'):8.1f} {m('loc_pix'):8.1f} "
              f"{m('jud_pix'):8.1f}")

    # 跨类:GT 面积 vs 档位增益(L1→full 的 Δ判定img / Δ判定pix)
    xs, ys_img, ys_pix = [], [], []
    for cls, r in out["results"].items():
        xs.append(r["gt_area_mean"])
        ys_img.append(r["tiers"]["full"]["jud_img"] - r["tiers"]["L1"]["jud_img"])
        ys_pix.append(r["tiers"]["full"]["jud_pix"] - r["tiers"]["L1"]["jud_pix"])
    out["cross_class_area_vs_dimg"] = [round(float(v), 4) for v in
                                       spearmanr(xs, ys_img)]
    out["cross_class_area_vs_dpix"] = [round(float(v), 4) for v in
                                       spearmanr(xs, ys_pix)]
    print(f"\n跨类 Spearman(GT面积, Δ判定img) = {out['cross_class_area_vs_dimg']}")
    print(f"跨类 Spearman(GT面积, Δ判定pix) = {out['cross_class_area_vs_dpix']}")
    print("\n逐类 GT面积 / 三档(按面积升序):")
    for cls, r in sorted(out["results"].items(),
                         key=lambda kv: kv[1]["gt_area_mean"]):
        t = r["tiers"]
        print(f"  {cls:12s} 面积 {r['gt_area_mean']:.3f}  "
              f"判定img {t['L1']['jud_img']:5.1f}→{t['L1+3x3']['jud_img']:5.1f}"
              f"→{t['full']['jud_img']:5.1f}  "
              f"判定pix {t['L1']['jud_pix']:5.1f}→{t['L1+3x3']['jud_pix']:5.1f}"
              f"→{t['full']['jud_pix']:5.1f}  "
              f"定位pix {t['L1']['loc_pix']:5.1f}→{t['full']['loc_pix']:5.1f}  "
              f"类内rho {r['within_class_rho_area_vs_gain']}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
