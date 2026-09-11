"""建**良品**特征缓存 —— 过杀率与阈值标定只能来自这里。

为什么单独一个脚本(而不是复用 cache_features):后者里 `if mp is None:
continue` 把无掩膜的良品图全部跳过,导致缓存里只有缺陷图,过杀率无从计算。
产线的阈值协议是"从不含缺陷的样本标定",良品缓存是必需项。

与共享/cache_good.py 同算法,两处移植差异(为接入本仓库的运行时):
  - 引擎经 build_engine 按目录自动选后端(原版硬编码 OVEngine)
  - 走 engine.window_indices 取窗索引(原版直接从 data/deploy 读 .npy)
  - 支持 --classes all 与幂等续跑

缓存 full/w3/w5 原始特征(few 分数评估时现算,纯矩阵乘很便宜)+ susp。

**susp 也要存**:过杀率要求良品走与缺陷品**完全同一条**选窗链路
(cv 策略下 s3/s5 由 susp 决定)。良品不参与选窗是把阈值定在良品上、
不拿缺陷调参;但良品的图分数必须按同样算法算出来,否则过杀率没有意义。

用法:
    python scripts/cache_good.py --data_root /root/autodl-tmp/mvtec_anomaly_detection \
        --deploy data/deploy_onnx_dyn --classes all --out /root/autodl-tmp/feat_cache_good
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                              # noqa: E402
from mvtec import MVTEC_CLASSES                           # noqa: E402
from runtime.line.config import CvCfg                     # noqa: E402
from runtime.line.preprocess_cv import (                  # noqa: E402
    normalize_illumination, patch_suspicion, suspicion_map)
from runtime.pipeline import GRID                         # noqa: E402
from scripts.eval_ov import build_engine                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--deploy", default="data/deploy_onnx_dyn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classes", default="all")
    ap.add_argument("--out", default="/tmp/feat_cache_good")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="缓哪个良品划分。test = 评估用良品(阈值/校准来源);"
                         "train = few-shot gallery 池。两者角色不同,必须分开缓存 ——"
                         "拿评估良品当 gallery 是自引用。")
    ap.add_argument("--class_table", default="auto", choices=["auto", "mvtec", "4i"],
                    help="--classes all 时用哪张类表。auto = 看 data_root 里有没有 4i")
    args = ap.parse_args()

    if args.classes == "all":
        tbl = args.class_table
        if tbl == "auto":
            tbl = "4i" if "4i" in str(args.data_root) else "mvtec"
        if tbl == "4i":
            # 从**数据目录**扫类名,不查硬编码表 —— 权威集是
            # data/surface_defects_4i(make_4i_mvtec.py 产物)
            from classes4i import classes as c4_classes
            classes = c4_classes(args.data_root)
        else:
            classes = MVTEC_CLASSES
    else:
        classes = [c.strip() for c in args.classes.split(",")]
    eng, backend, device = build_engine(args.deploy, args.device)
    print(f"[engine] {backend} | {device}", flush=True)

    root = Path(args.data_root)
    out = Path(args.out)
    t0 = time.time()
    for cls in classes:
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        if (d / "done").exists():
            n = len(list(d.glob("[0-9]*.npz")))
            print(f"[skip] {cls} ({n} 张已缓存)", flush=True)
            continue
        idx3, idx2 = eng.window_indices(3), eng.window_indices(2)
        n = 0
        if args.split == "train":
            # train/good 全是正常样本,没有 rel/掩膜,直接列文件
            paths = mvtec.iter_train_images(root, cls)
        else:
            paths = [ip for _n, rel, ip, _m in mvtec.iter_test_images(root, cls)
                     if rel == "good"]
        for ip in paths:
            f = d / f"{n:03d}.npz"
            if not f.exists():
                x = eng.preprocess_rgb(
                    np.asarray(Image.open(ip).convert("RGB")))
                toks = eng.patcher(x)
                full = eng.tower_full(toks)
                seq3 = np.concatenate(
                    [np.zeros((idx3.shape[0], 1), np.int64), idx3], axis=1)
                w3 = eng.tower_w(toks[0][seq3])[:, 0]
                seq2 = np.concatenate(
                    [np.zeros((idx2.shape[0], 1), np.int64), idx2], axis=1)
                w5 = eng.tower_w(toks[0][seq2])[:, 0]
                g = np.asarray(Image.open(ip).convert("L"))
                cv = CvCfg(illum_norm="clahe")
                sus = patch_suspicion(
                    suspicion_map(normalize_illumination(g, cv), cv), GRID)
                np.savez(f, full=full[0], w3=w3, w5=w5, susp=sus)
            n += 1
            if n % 25 == 0:
                print(f"  [{cls}] good {n}", flush=True)
        (d / "done").touch()
        print(f"[done] {cls} ({n} 张良品) {time.time()-t0:.0f}s", flush=True)

    print(f"\n[all] 良品缓存完成 → {out} | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
