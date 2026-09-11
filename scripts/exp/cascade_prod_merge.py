"""级联的**判定口径**复核:融合规则 × 图像分数公式,全 15 类全量。

## 为什么必须单独跑这一遍

任务书指定的 `共享/cascade_v2.py` 与产线 `runtime/pipeline.py` 有两处**实质
不同**,不查清就不能拿它的 img AUROC 当"产线精度":

| | 共享/cascade_v2.py | 产线 pipeline.py(few 分支) |
|---|---|---|
| 窗口并入地基 | **调和**平均 `1/max(base,1e-12)` | **算术**平均 `num/den` |
| 图像分数 | `refined.max()` | `(cls_prob + few_map.max())/2` |

本地 v1 报告已实测:**调和平均被小值主导**,会把地基拉垮
(tile 92.5 → 57.0,而 max 融合 = 92.5 与地基逐位相同)。若我的主结果用的是
调和平均,那"窗口塔有害"可能只是**融合规则的锅**,不是窗口塔本身的性质 ——
两者结论完全不同,必须分开。

图像分数同理:本地 v3 报告明确区分
  **定位口径** = few 分支单独(不含任何全局标量)→ 缺陷在**哪里**
  **判定口径** = `(cls_prob + few_map.max())/2`             → 整件**是不是** NG
产线的 OK/NG 用后者。拿 `max()` 当判定分数,等于把定位口径的数字
当成产线精度报出去 —— 正是 v2 犯过的错。

## 本脚本测什么

对每个 (预算 B, 选窗策略 st, 融合规则 rule, 图像分数口径口径):
  预算 0/8/16/32/64/128/999(999=全量窗)
  策略 none/cv/fewshot/rand
  融合 harmonic / arithmetic
  分数 loc(`few_map.max()`) / prod(`(cls_prob + few_map.max())/2`)

只算 img-level(主指标);像素 AUROC 在主网格里已有,不重复。

输出:每类的 img AUROC + 过杀@P99 + 漏检@P99(阈值只从良品标定)。

用法:
    python scripts/exp/cascade_prod_merge.py --out /tmp/cascade_prod.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/autodl-tmp/winclip")
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P   # noqa: E402
from scripts.fast_upsample import upsample_bilinear_fast      # noqa: E402

CD = Path("/root/autodl-tmp/feat_cache_gpu")
CG = Path("/root/autodl-tmp/feat_cache_good")
TD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn/text_protos")
DD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn")
BUDGETS = [0, 8, 16, 32, 64, 128, 999]
STRATS = ["none", "cv", "fewshot", "rand"]
RULES = ["harmonic", "arithmetic"]
few = P._few_token_score
CLASSES = ["tile", "carpet", "bottle", "metal_nut", "screw", "cable", "capsule",
           "grid", "hazelnut", "leather", "pill", "toothbrush", "transistor",
           "wood", "zipper"]


def precompute(f: Path, gal: dict, pos, neg, temp):
    """一图一次:把与预算/策略无关的量全算好(这是本脚本能跑全量的前提)。"""
    z = np.load(f)
    full = z["full"]
    base = few(full[1:], gal["patch"])                     # (225,)
    f3 = few(z["w3"], gal["large"])                        # (169,)
    f5 = few(z["w5"], gal["mid"])                          # (196,)
    # zero 分支的窗口概率也要:产线 m_all 的地基是 cls_prob,
    # 窗口以**调和**平均并入 zero 分支(见 pipeline.anomaly_maps)
    z3 = P._prob(z["w3"], pos, neg, temp)
    z5 = P._prob(z["w5"], pos, neg, temp)
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])  # 标量
    return dict(base=base, f3=f3, f5=f5, z3=z3, z5=z5,
                cls_prob=cls_prob,
                susp=z["susp"] if "susp" in z.files else None,
                gt=z["gt"] if "gt" in z.files else None)


def merge(pre, B, st, rule, idx3, idx2):
    """→ (225,) few_map。未选中区域保留地基(任务书修正二)。"""
    base = pre["base"]
    if B == 0 or st == "none":
        return base
    if st == "cv":
        s3 = pre["susp"].ravel()[idx3 - 1].mean(axis=1)
        s5 = pre["susp"].ravel()[idx2 - 1].mean(axis=1)
    elif st == "fewshot":
        s3, s5 = pre["f3"], pre["f5"]
    else:
        r = np.random.default_rng(0)
        s3, s5 = r.random(idx3.shape[0]), r.random(idx2.shape[0])
    n3 = min(B, idx3.shape[0])
    n5 = min(max(2, int(B * idx2.shape[0] / idx3.shape[0])), idx2.shape[0])
    q3 = np.argsort(-s3, kind="stable")[:n3]
    q5 = np.argsort(-s5, kind="stable")[:n5]

    if rule == "harmonic":
        # 共享脚本的写法:1/(m),对 0 夹 1e-12
        inv = 1.0 / np.maximum(base, 1e-12)
        cnt = np.ones(N_PATCH, np.float32)
        for f, q, idx in ((pre["f3"], q3, idx3), (pre["f5"], q5, idx2)):
            m, c = P._scatter_harmonic(f[q], idx[q])
            pr = c > 0
            inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
            cnt[pr] += 1.0
        return cnt / inv
    # arithmetic:产线 few 分支的写法(num/den,每尺度计数归一)
    num = base.copy()
    den = np.ones(N_PATCH, np.float32)
    for f, q, idx in ((pre["f3"], q3, idx3), (pre["f5"], q5, idx2)):
        m, c = P._scatter_harmonic(f[q], idx[q])
        pr = c > 0
        num[pr] += m[pr]
        den[pr] += 1.0
    return num / den


def prod_map(pre, B, st, idx3, idx2):
    """产线判定的**像素图**:`m_all = harmon(cls_prob, 窗口zero) + few_map`。

    逐行镜像 pipeline.anomaly_maps 的两段:
      zero 分支:inv 从 1/cls_prob 起,每个已覆盖尺度加 1/m → 调和平均
      few  分支:num/den 算术平均,再与 zero 分支**相加**
    未选中的尺度不进 inv/den(等价于该 patch 只有 cls_prob 与 patch few 分)。
    """
    few_map = merge(pre, B, st, "arithmetic", idx3, idx2)
    if B == 0 or st == "none":
        # 无窗口:zero 分支整幅恒为 cls_prob
        return np.full(N_PATCH, pre["cls_prob"], np.float32) + few_map
    # 选窗(与 merge 用同一套排序,保证两处选中的是同一批窗)
    if st == "cv":
        s3 = pre["susp"].ravel()[idx3 - 1].mean(axis=1)
        s5 = pre["susp"].ravel()[idx2 - 1].mean(axis=1)
    elif st == "fewshot":
        s3, s5 = pre["f3"], pre["f5"]
    else:
        r = np.random.default_rng(0)
        s3, s5 = r.random(idx3.shape[0]), r.random(idx2.shape[0])
    n3 = min(B, idx3.shape[0])
    n5 = min(max(2, int(B * idx2.shape[0] / idx3.shape[0])), idx2.shape[0])
    q3 = np.argsort(-s3, kind="stable")[:n3]
    q5 = np.argsort(-s5, kind="stable")[:n5]

    inv = np.full(N_PATCH, 1.0 / pre["cls_prob"], np.float32)
    n_terms = np.ones(N_PATCH, np.float32)
    for zp, q, idx in ((pre["z3"], q3, idx3), (pre["z5"], q5, idx2)):
        m, c = P._scatter_harmonic(zp[q], idx[q])
        pr = c > 0
        inv[pr] += 1.0 / m[pr]
        n_terms[pr] += 1.0
    return (n_terms / inv).astype(np.float32) + few_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/cascade_prod.json")
    ap.add_argument("--classes", default="all")
    args = ap.parse_args()
    classes = CLASSES if args.classes == "all" else args.classes.split(",")
    idx3 = np.load(DD / "win_idx_k3.npy")
    idx2 = np.load(DD / "win_idx_k2.npy")

    out = {"budgets": BUDGETS, "strategies": STRATS, "rules": RULES,
           "results": {}}
    t0 = time.time()
    for cls in classes:
        d = np.load(TD / f"{cls}.npz")
        pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
        gal = dict(np.load(CD / cls / "gallery.npz"))
        bad, good = [], []
        for f in sorted((CD / cls).glob("[0-9]*.npz")):
            p = precompute(f, gal, pos, neg, temp)
            if p["gt"] is not None and p["gt"].any():
                bad.append(p)
        for f in sorted((CG / cls).glob("[0-9]*.npz")):
            good.append(precompute(f, gal, pos, neg, temp))

        res = {}
        for B in BUDGETS:
            for st in STRATS:
                if B == 0 and st != "none":
                    continue
                zeros_good, zeros_bad, ones_good, ones_bad = [], [], [], []
                pix, pgt = [], []
                for tag, recs in (("g", good), ("b", bad)):
                    for p in recs:
                        m = merge(p, B, st, "arithmetic", idx3, idx2)
                        # 产线判定分数 = (cls_prob + few_map.max())/2
                        s = p["cls_prob"] if B == 0 or st == "none" else \
                            (p["cls_prob"] + float(m.max())) / 2.0
                        (zeros_good if tag == "g" else zeros_bad).append(s)
                        # 定位口径 = 纯 few_map.max(),不含任何全局标量
                        (ones_good if tag == "g" else ones_bad).append(
                            float(m.max()))
                        if tag == "b":
                            mp = prod_map(p, B, st, idx3, idx2)
                            pix.append(upsample_bilinear_fast(
                                mp.reshape(GRID, GRID), 240, 240).flatten())
                            pgt.append(p["gt"].flatten())
                zg, zb = np.array(zeros_good), np.array(zeros_bad)
                lg, lb = np.array(ones_good), np.array(ones_bad)
                row = {"n_good": len(zg), "n_bad": len(zb),
                       "img_auc_prod": round(float(roc_auc_score(
                           np.r_[np.zeros(len(zg)), np.ones(len(zb))],
                           np.r_[zg, zb])) * 100, 1),
                       "img_auc_loc": round(float(roc_auc_score(
                           np.r_[np.zeros(len(lg)), np.ones(len(lb))],
                           np.r_[lg, lb])) * 100, 1),
                       "pix_auc_prod": round(float(roc_auc_score(
                           np.concatenate(pgt), np.concatenate(pix))) * 100, 1)}
                thr = float(np.percentile(zg, 99))
                row["overkill_p99"] = round(float((zg > thr).mean()) * 100, 1)
                row["escape_p99"] = round(float((zb <= thr).mean()) * 100, 1)
                res[f"{B}|{st}"] = row
        out["results"][cls] = res
        print(f"[done] {cls}  良品 {len(good)} / 缺陷 {len(bad)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[out] {args.out}")

    # 15 类均值对照
    print(f"\n{'策略':8s} {'B':>4s} {'判定img':>9s} {'定位img':>9s} "
          f"{'判定pix':>9s} {'过杀':>7s} {'漏检':>7s}")
    for st in STRATS:
        for B in BUDGETS:
            k = f"{B}|{st}"
            rows = [r[k] for r in out["results"].values() if k in r]
            if not rows:
                continue
            f = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
            print(f"{st:8s} {B:4d} {f('img_auc_prod'):9.1f} "
                  f"{f('img_auc_loc'):9.1f} {f('pix_auc_prod'):9.1f} "
                  f"{f('overkill_p99'):6.1f}% {f('escape_p99'):6.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
