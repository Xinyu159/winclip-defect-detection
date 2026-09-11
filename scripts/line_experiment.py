"""产线级联实验:精度 vs 算力预算,三种选区策略对照。

回答的问题:把窗口预算从全量压到 8/16/32/64,精度掉多少?掉的部分
能不能用传统 CV 先验补回来?——这是级联架构成立的前提,也是简历数字的来源。

三种选区策略(都不许碰 GT):
    full      全量窗口(上界参考)
    saliency  用 patch 文本分选窗(免费,来自整图粗筛复用)
    cvroi     用传统 CV 可疑度选窗(零神经算力)
    random    随机选(下界对照,证明"预算本身"值多少、选窗策略值多少)

协议与 evaluate.py / eval_ov.py 一致:seed42 采样 gallery,GT resize 到 240
阈值 128,pixel AUROC 只统计有 GT 的缺陷图。产出 logs/line_exp_*.json。

用法:
    python scripts/line_experiment.py --classes tile,carpet --n 25 --shots 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                              # noqa: E402
from mvtec import MVTEC_CLASSES                           # noqa: E402
from runtime.line.preprocess_cv import (                  # noqa: E402
    normalize_illumination, patch_suspicion, suspicion_map)
from runtime.line.config import CvCfg                     # noqa: E402
from runtime.pipeline import GRID, N_PATCH, OVPipeline    # noqa: E402
from scripts.eval_ov import build_engine, upsample_bilinear_np   # noqa: E402


def cache_features(eng, pipe, root, cls, n_img, shots, cache_dir: Path):
    """缓存每图的全部特征,后续扫预算免重算(塔前向是大头)。"""
    d = cache_dir / cls
    d.mkdir(parents=True, exist_ok=True)
    if (d / "done").exists():
        return
    rng = np.random.default_rng(42)
    tr = mvtec.iter_train_images(root, cls)
    picks = [tr[i] for i in rng.choice(len(tr), size=shots, replace=False)]
    gal = np.concatenate([eng.preprocess_rgb(
        np.asarray(Image.open(p).convert("RGB"))) for p in picks], axis=0)
    pipe.set_gallery(gal)
    np.savez(d / "gallery.npz", **pipe.gallery)

    i = 0
    for _, rel, ip, mp in mvtec.iter_test_images(root, cls):
        if mp is None or i >= n_img:
            continue
        f = d / f"{i:03d}.npz"
        # 缓存完整性:五个字段缺一不可。上一版漏写 susp,导致打分阶段
        # KeyError;缓存复用必须校验字段而不是只看文件在不在。
        need = {"full", "w3", "w5", "gt", "susp"}
        if not (f.exists() and need <= set(np.load(f).files)):
            g = np.asarray(Image.open(ip).convert("L"))
            x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert("RGB")))
            toks = eng.patcher(x)
            full = eng.tower_full(toks)
            w3 = pipe._window_feats(toks, 3, None)
            w5 = pipe._window_feats(toks, 2, None)
            gt = np.asarray(Image.open(mp).convert("L")
                            .resize((240, 240), Image.BILINEAR)) > 128
            cv = CvCfg(illum_norm="clahe")
            gn = normalize_illumination(g, cv)
            sus = patch_suspicion(suspicion_map(gn, cv), GRID)
            np.savez(f, full=full[0], w3=w3, w5=w5, gt=gt, susp=sus)
        i += 1
        print(f"  [{cls}] cache {i}/{n_img}", flush=True)
    (d / "done").touch()


def score(eng, pipe, items, gal, cls, budget=None, strategy="full",
          text_dir=None):
    d = np.load(Path(text_dir or "data/deploy/text_protos") / f"{cls}.npz")
    pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
    P, harm = OVPipeline._prob, OVPipeline._scatter_harmonic
    few = OVPipeline._few_token_score
    idx3, idx5 = eng.window_indices(3), eng.window_indices(2)
    cen3 = idx3[:, 4] - 1
    cen5 = idx5 - 1

    ps, pg = [], []
    for it in items:
        full, w3, w5, gt, susp = (it["full"], it["w3"], it["w5"],
                                  it["gt"], it["susp"])
        if not gt.any():
            continue
        cls_p = float(P(full[:1], pos, neg, temp)[0])
        patch_p = P(full[1:], pos, neg, temp)

        sel = {}
        for key, nw, idx in (("w3", 169, idx3), ("w5", 196, idx5)):
            if budget is None:
                sel[key] = np.arange(nw)
                continue
            nb = budget if key == "w3" else max(2, int(budget * 196 / 169))
            nb = min(nb, nw)
            if strategy == "random":
                sel[key] = np.random.default_rng(0).choice(nw, nb, replace=False)
            elif strategy == "saliency":
                sc = np.array([patch_p[i - 1].mean() for i in idx])
                sel[key] = np.argsort(-sc, kind="stable")[:nb]
            elif strategy == "cvroi":
                su = susp.ravel()
                sc = np.array([su[i - 1].mean() for i in idx])
                sel[key] = np.argsort(-sc, kind="stable")[:nb]
            else:
                sel[key] = np.arange(nw)

        parts = {}
        for key, feats, idx in (("w3", w3, idx3), ("w5", w5, idx5)):
            s = sel[key]
            parts[key] = harm(P(feats[s], pos, neg, temp), idx[s])

        m48, c48 = parts["w3"]; m32, c32 = parts["w5"]
        inv = np.full(N_PATCH, 1.0 / max(cls_p, 1e-12), np.float32)
        nt = np.ones(N_PATCH, np.float32)
        for m, c in ((m48, c48), (m32, c32)):
            pr = c > 0
            inv[pr] += 1.0 / m[pr]; nt[pr] += 1.0
        m_all = nt / inv

        if gal is not None:
            pm = few(full[1:], gal["patch"])
            num = pm.copy(); den = np.ones(N_PATCH, np.float32)
            for key, feats, idx, gn in (("w3", w3, idx3, "large"),
                                        ("w5", w5, idx5, "mid")):
                s = sel[key]
                m, c = harm(few(feats[s], gal[gn]), idx[s])
                pr = c > 0
                num[pr] += m[pr]; den[pr] += 1.0
            m_all = m_all + num / den

        up = upsample_bilinear_np(m_all.reshape(GRID, GRID), 240, 240)
        ps.append(up.flatten()); pg.append(gt.flatten())
    if not pg:
        return float("nan")
    return round(float(roc_auc_score(np.concatenate(pg),
                                     np.concatenate(ps)) * 100), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--classes", default="tile,carpet,bottle,metal_nut,screw",
                    help="逗号分隔,或 'all'(MVTec 15 类)")
    ap.add_argument("--n", type=int, default=25,
                    help="每类缺陷图张数;0 = 不限(全部)")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--cache", default="/tmp/feat_cache")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--strategies", default="full,saliency,cvroi,random")
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.classes == "all":
        args.classes = ",".join(MVTEC_CLASSES)
    root = Path(args.data_root)
    classes = [c.strip() for c in args.classes.split(",")]
    cache_dir = Path(args.cache)
    eng, backend, device = build_engine(args.deploy, args.device)
    pipe = OVPipeline(eng, args.text)
    # n=0 表示"不限张数"。cache_features 里用的是 i >= n_img 判断,0 会让
    # 第一张就被截掉,所以在这里换成一个大数(而不是让调用方自己换算)。
    n_img = 10**9 if args.n == 0 else args.n
    print(f"[engine] {backend} | device={device} | 类数 {len(classes)} | "
          f"n={args.n if args.n else 'all'}", flush=True)

    print("[1/2] 建特征缓存(塔前向只跑一次)", flush=True)
    for cls in classes:
        pipe.set_class(cls)
        cache_features(eng, pipe, root, cls, n_img, args.shots, cache_dir)

    print("\n[2/2] 扫描预算 × 策略", flush=True)
    results = {"meta": {"classes": classes, "n_img": args.n,
                        "shots": args.shots, "engine": backend,
                        "device": device, "deploy": args.deploy,
                        "tag": args.tag,
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
               "results": {}}
    strategies = args.strategies.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]

    for cls in classes:
        d = cache_dir / cls
        g = np.load(d / "gallery.npz")
        gal = {k: g[k] for k in g.files} if args.shots > 0 else None
        items = [np.load(p) for p in sorted(d.glob("[0-9]*.npz"))]
        row = {}
        for strat in strategies:
            bset = [None] if strat == "full" else budgets
            for b in bset:
                key = strat if b is None else f"{strat}_{b}"
                t0 = time.time()
                row[key] = score(eng, pipe, items, gal, cls,
                                 budget=b, strategy=strat,
                                 text_dir=args.text)
                print(f"  {cls:10s} {key:12s} pix={row[key]:6.1f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        results["results"][cls] = row
        Path("logs").mkdir(exist_ok=True)
        (Path("logs") / "line_exp_latest.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== 汇总(各类均值)===")
    keys = list(results["results"][classes[0]].keys())
    for k in keys:
        v = np.mean([results["results"][c][k] for c in classes])
        print(f"  {k:14s} {v:6.1f}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out = Path("logs") / f"line_exp_{ts}{suffix}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[log] {out}")


if __name__ == "__main__":
    main()
