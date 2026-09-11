"""级联 v2 实验 —— 修正架构后的精度/算力曲线(全 15 类)。

移植自 `远程任务/共享/cascade_v2.py`(本地在 5 类×25 张上已验证),三处差异:
  1. 后端经 build_engine 自动选(ONNX/ORT 或 OpenVINO),不硬编码
  2. 路径可配,默认接 line_experiment 已建好的全量缺陷缓存(免重跑塔前向)
  3. 上采样走向量化实现,且**启动时强制与参考实现逐位对拍**(见 fast_upsample)

被修正的 v1 架构性错误(来自本地报告 results/cascade_tier_report.md):
  1. 用 `_prob(full[1:], 文本原型)` 选窗 —— 实测与 GT **反相关**
     (5 类 pixel AUROC 14.3/7.3/12.7/19.3/16.2)。整图 mosaic 令牌是
     patch 尺度的聚合特征,不在文本原型对齐的空间里。
  2. 未覆盖 patch 填 `cls_prob` —— 缺陷图上 CLS 高 → **整片背景被抬到
     比缺陷还高**(实测 GT 内 0.126 / GT 外 0.255,完全倒挂)。
  3. 缺良品缓存 → 过杀率无从算起(阈值必须从良品标定)。

正确架构:
  base   = few(mosaic_patch, gallery.patch)          # 225 值,免费(L1 已产出)
  refine = 选中窗口的 few 窗口分,按调和平均并入 base
  未选中区域:保留 base 值

四种选窗策略(全部不碰 GT):
  none     不精检(B=0,纯地基,作为上界参照之一)
  cv       L0 传统 CV 可疑度(零神经算力)
  fewshot  用 L1 的 patch few 分(免费,来自已算特征)
  rand     随机(下界对照)

用法:
    python scripts/cascade_experiment.py --stage scan --classes all \
        --budgets 0,8,16,32,64,128 --shots 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from mvtec import MVTEC_CLASSES                           # noqa: E402
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P   # noqa: E402
from scripts.eval_ov import build_engine                  # noqa: E402
from scripts.fast_upsample import (                       # noqa: E402
    upsample_bilinear_fast, verify_fast_upsample)

#: 999 = **不设上限**,即取满全部 169(3×3)+196(2×2) 窗 = research 完整路径。
#: 任务书 §1.5 的核心待验证问题(L1 与 L1+3×3 是否仍然逐位相同)需要一个
#: "全量塔"锚点才能回答:拿 none|0(纯 L1)与 fewshot|999(全量塔)比,
#: 若逐位相同,则比"L1+3×3"更强的结论都成立(全量都不加,部分更不会加)。
BUDGETS = (0, 8, 16, 32, 64, 128, 999)
STRATS = ("none", "cv", "fewshot", "rand")


def load_split(CD: Path, CG: Path, cls: str, good: bool = False):
    """读缓存。缺陷侧校验 gt 字段;两侧共用 gallery(缺陷缓存的 gallery)。

    注意:良品缓存目录里没有 gallery.npz(见 cache_good.py),gallery 统一
    从缺陷缓存目录取 —— 两边必须是**同一个 gallery**,否则地基尺度不一致,
    过杀率就没意义了。
    """
    root = CG / cls if good else CD / cls
    gal = dict(np.load(CD / cls / "gallery.npz"))
    fs = sorted(f for f in root.glob("[0-9]*.npz") if f.name != "gallery.npz")
    recs = []
    for f in fs:
        z = np.load(f)
        if good:
            recs.append(dict(gt_img=False, full=z["full"], w3=z["w3"],
                             w5=z["w5"], susp=z["susp"]))
        else:
            gt = z["gt"]
            if not gt.any():        # 无掩膜=无缺陷,不属于缺陷集
                continue
            recs.append(dict(gt_img=True, gt=gt, full=z["full"], w3=z["w3"],
                             w5=z["w5"], susp=z["susp"]))
    return gal, recs


def score_map(r, gal, B, st, idx3, idx2):
    """→ (225 长 map, 图像分数)。

    地基 base 恒为 patch 尺度 few 分:零额外 token(全部来自 L1 已算的
    full 特征),且天然覆盖全画幅 —— 未选中的区域保留它,而不是记 0/填 CLS。
    """
    few, harm = P._few_token_score, P._scatter_harmonic
    base = few(r["full"][1:], gal["patch"])           # 免费地基
    if B == 0 or st == "none":
        return base, float(base.max())

    f3 = few(r["w3"], gal["large"])
    f5 = few(r["w5"], gal["mid"])
    if st == "cv":
        if r["susp"] is None:
            raise AssertionError("CV 选窗需要可疑度图;两边缓存都要含 susp")
        s3 = r["susp"].ravel()[idx3 - 1].mean(axis=1)
        s5 = r["susp"].ravel()[idx2 - 1].mean(axis=1)
    elif st == "fewshot":
        s3, s5 = f3, f5
    else:
        s3 = np.random.default_rng(0).random(idx3.shape[0])
        s5 = np.random.default_rng(1).random(idx2.shape[0])
    n5 = min(idx2.shape[0], max(2, int(B * idx2.shape[0] / idx3.shape[0])))
    q3 = np.argsort(-s3, kind="stable")[:B]
    q5 = np.argsort(-s5, kind="stable")[:n5]

    # 精检窗:窗口分与地基调和平均(窗口分是更强的局部证据时才会胜出)
    m3, c3 = harm(f3[q3], idx3[q3])
    m5, c5 = harm(f5[q5], idx2[q5])
    inv = 1.0 / np.maximum(base, 1e-12)
    cnt = np.ones(N_PATCH, np.float32)
    for m, c in ((m3, c3), (m5, c5)):
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        cnt[pr] += 1.0
    refined = cnt / inv
    return refined, float(refined.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="scan", choices=["scan"])
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--deploy", default="data/deploy_onnx_dyn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classes", default="all")
    ap.add_argument("--cache", default="/tmp/feat_cache", help="缺陷缓存目录")
    ap.add_argument("--cache-good", default="/tmp/feat_cache_good")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS),
                    help="999 = 全部窗口(不设上限),作为 research 完整路径锚点")
    ap.add_argument("--strategies", default=",".join(STRATS))
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    classes = MVTEC_CLASSES if args.classes == "all" else \
        [c.strip() for c in args.classes.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    strats = [s.strip() for s in args.strategies.split(",")]
    CD, CG = Path(args.cache), Path(args.cache_good)

    # 上采样必须与参考实现一致才继续:AUROC 对排序敏感,快版本错一点点
    # 就可能换序,而换序是静默的(数字看着正常)。
    d = verify_fast_upsample()
    if d > 1e-12:
        print(f"[FAIL] 向量化上采样与参考不一致 max|Δ|={d:.2e},拒绝继续")
        return 1

    # 窗口索引经引擎取(两后端同源),不从某个固定目录读
    eng, backend, device = build_engine(args.deploy, args.device)
    idx3, idx2 = eng.window_indices(3), eng.window_indices(2)
    print(f"[engine] {backend} | {device} | 窗索引 "
          f"k3={idx3.shape} k2={idx2.shape}", flush=True)

    allres = {"meta": {"classes": classes, "budgets": budgets,
                       "strategies": strats, "shots": args.shots,
                       "backend": backend, "device": device,
                       "cache": str(CD), "cache_good": str(CG),
                       "tag": args.tag,
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
              "results": {}}
    t0 = time.time()
    for cls in classes:
        gal, bad = load_split(CD, CG, cls, good=False)
        try:
            _, good = load_split(CD, CG, cls, good=True)
        except FileNotFoundError:
            print(f"[{cls}] 良品缓存缺失,跳过", flush=True)
            continue
        res = {}
        for B in budgets:
            for st in strats:
                if B == 0 and st != "none":
                    continue
                ps, pg, gs, bs = [], [], [], []
                for r in bad:
                    m, s = score_map(r, gal, B, st, idx3, idx2)
                    up = upsample_bilinear_fast(
                        m.reshape(GRID, GRID), 240, 240)
                    ps.append(up.flatten())
                    pg.append(r["gt"].flatten())
                    bs.append(s)
                for r in good:
                    _, s = score_map(r, gal, B, st, idx3, idx2)
                    gs.append(s)
                gs, bs = np.array(gs), np.array(bs)
                row = {"n_good": len(gs), "n_bad": len(bs),
                       "pix_auc": round(float(roc_auc_score(
                           np.concatenate(pg), np.concatenate(ps))) * 100, 1)}
                if len(gs):
                    row["img_auc"] = round(float(roc_auc_score(
                        np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                        np.r_[gs, bs])) * 100, 1)
                    for tag, q in (("p95", 95), ("p99", 99)):
                        thr = float(np.percentile(gs, q))
                        row[f"overkill_{tag}"] = round(
                            float((gs > thr).mean()) * 100, 1)
                        row[f"escape_{tag}"] = round(
                            float((bs <= thr).mean()) * 100, 1)
                res[f"{B}|{st}"] = row
        allres["results"][cls] = res
        print(f"[done] {cls}  良品 {len(good)} / 缺陷 {len(bad)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        # 逐类落盘:全量跑要几十分钟,中途断了不该丢掉已算好的类
        Path("logs").mkdir(exist_ok=True)
        Path("logs/cascade_v2_latest.json").write_text(
            json.dumps(allres, ensure_ascii=False, indent=1))

    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(args.out or f"logs/cascade_v2_{ts}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print(f"\n[log] {out}")

    # 控制台汇总
    print(f"\n{'类':12s} {'策略':8s} {'B':>4s} {'pixAUROC':>9s} "
          f"{'imgAUROC':>9s} {'过杀@P99':>9s} {'漏检@P99':>9s}")
    for cls, r in allres["results"].items():
        for B in budgets:
            for st in strats:
                k = f"{B}|{st}"
                if k not in r:
                    continue
                d = r[k]
                print(f"{cls:12s} {st:8s} {B:4d} {d['pix_auc']:9.1f} "
                      f"{d.get('img_auc', float('nan')):9.1f} "
                      f"{d.get('overkill_p99', float('nan')):8.1f}% "
                      f"{d.get('escape_p99', float('nan')):8.1f}%")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
