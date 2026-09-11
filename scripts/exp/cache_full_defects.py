"""把缺陷图缓存扩到**全量** —— 用于验证"子集假设"。

动机:报告里 metal_nut 与 research 差 28 个像素点(89.5 vs 61.5),
我推测原因是"测试子集不同"(本报告只用前 25 张)。但那是**推测,没有证据**。
本脚本建全量缓存,让这个推测要么被证实、要么被推翻。

同时这也是对核心结论更强的检验:L1 与 L1+3×3 的关系是否在
**全部**缺陷图上仍然成立,而不是只在前 25 张上成立。

读的是 MVTec 布局,故 **data/surface_defects_4i 也走同一个脚本**
(make_4i_mvtec.py 就是转成这个布局的;缺陷 240px 掩膜阈值 >128,
该集在转换期已把 GT 二值化到 {0,255},故 >128 是恒等)。

用法:
    python scripts/exp/cache_full_defects.py --classes metal_nut,screw
    # 4i(远程 GPU):
    python scripts/exp/cache_full_defects.py --data_root /root/autodl-tmp/surface_defects_4i \
        --deploy data/deploy_onnx_dyn --device cuda --classes all --cache /root/autodl-tmp/i4_defects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mvtec                                              # noqa: E402
from scripts.eval_ov import build_engine                  # noqa: E402
from runtime.pipeline import GRID                         # noqa: E402
from runtime.line.preprocess_cv import (                  # noqa: E402
    normalize_illumination, patch_suspicion, suspicion_map)
from runtime.line.config import CvCfg                     # noqa: E402

RS = getattr(Image, "Resampling", Image).BILINEAR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--classes", default="metal_nut,screw")
    ap.add_argument("--cache", default="/tmp/feat_cache")
    ap.add_argument("--device", default="cpu",
                    help="cpu=OpenVINO(本地) / cuda=ONNX-GPU(远程)。"
                         "经 build_engine 自动选后端 —— 早先这里硬编码 OVEngine,"
                         "远程要跑 CUDA 得改代码,故统一到与 cache_good.py 同一条路。")
    ap.add_argument("--class_table", default="auto", choices=["auto", "mvtec", "4i"],
                    help="--classes all 时用哪张类表。auto = 看 data_root 里有没有 4i")
    args = ap.parse_args()

    if args.classes == "all":
        tbl = args.class_table
        if tbl == "auto":
            tbl = "4i" if "4i" in str(args.data_root) else "mvtec"
        if tbl == "4i":
            from classes4i import classes as c4_classes
            classes = c4_classes(args.data_root)
        else:
            from mvtec import MVTEC_CLASSES as classes
    else:
        classes = args.classes.split(",")

    eng, backend, device = build_engine(args.deploy, args.device)
    print(f"[engine] {backend} | {device}", flush=True)
    idx3, idx2 = eng.window_indices(3), eng.window_indices(2)
    seq3 = np.concatenate([np.zeros((len(idx3), 1), np.int64), idx3], axis=1)
    seq2 = np.concatenate([np.zeros((len(idx2), 1), np.int64), idx2], axis=1)
    cv = CvCfg(illum_norm="clahe")

    for cls in classes:
        d = Path(args.cache) / cls
        d.mkdir(parents=True, exist_ok=True)
        need = {"full", "w3", "w5", "gt", "susp"}
        i = n_new = 0
        for _, rel, ip, mp in mvtec.iter_test_images(args.data_root, cls):
            if mp is None:
                continue                                  # 良品由 cache_good.py 负责
            f = d / f"{i:03d}.npz"
            i += 1
            if f.exists() and need <= set(np.load(f).files):
                continue                                  # 已有完整缓存,跳过
            x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert("RGB")))
            toks = eng.patcher(x)
            full = eng.tower_full(toks)
            w3 = eng.tower_w(toks[0][seq3])[:, 0]
            w5 = eng.tower_w(toks[0][seq2])[:, 0]
            gt = np.asarray(Image.open(mp).convert("L")
                            .resize((240, 240), RS)) > 128
            g = np.asarray(Image.open(ip).convert("L"))
            sus = patch_suspicion(
                suspicion_map(normalize_illumination(g, cv), cv), GRID)
            np.savez(f, full=full[0], w3=w3, w5=w5, gt=gt, susp=sus)
            n_new += 1
            print(f"  [{cls}] {i} (新增 {n_new})", flush=True)
        print(f"[done] {cls}: 共 {i} 张缺陷图,本次新增 {n_new}", flush=True)


if __name__ == "__main__":
    main()
