"""建**良品**特征缓存 —— 过杀率与阈值标定只能来自这里。

为什么单独一个脚本:原 cache_features 里 `if mp is None: continue` 把
无掩膜的良品图全部跳过,导致缓存里只有缺陷图,过杀率无从计算。
产线的阈值协议是"从不含缺陷的样本标定",所以良品缓存是必需项,不是可选项。

只缓存原始特征(full/w3/w5),few 分数在评估时现算(纯矩阵乘,很便宜)。
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import mvtec
from runtime.ov_engine import OVEngine
from runtime.line.preprocess_cv import normalize_illumination, suspicion_map, patch_suspicion
from runtime.line.config import CvCfg
from runtime.pipeline import GRID

DE = 'data/deploy'
ROOT = 'data/mvtec_anomaly_detection'
OUT = Path('/tmp/feat_cache_good')

ALL = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
       'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
       'wood', 'zipper']

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default=",".join(ALL))
ap.add_argument("--force", action="store_true", help="忽略 done 标记重跑")
args = ap.parse_args()

eng = OVEngine(DE, device='CPU')
idx3, idx2 = eng.window_indices(3), eng.window_indices(2)
seq3 = np.concatenate([np.zeros((len(idx3), 1), np.int64), idx3], axis=1)
seq2 = np.concatenate([np.zeros((len(idx2), 1), np.int64), idx2], axis=1)
cv = CvCfg(illum_norm="clahe")

for cls in args.classes.split(","):
    d = OUT / cls
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'done').exists() and not args.force:
        print(f"[skip] {cls}", flush=True)
        continue
    n = n_new = 0
    for name, rel, ip, mp in mvtec.iter_test_images(ROOT, cls):
        if rel != 'good':
            continue
        f = d / f"{n:03d}.npz"
        if not f.exists():
            x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert('RGB')))
            toks = eng.patcher(x)
            full = eng.tower_full(toks)
            w3 = eng.tower_w(toks[0][seq3])[:, 0]
            w5 = eng.tower_w(toks[0][seq2])[:, 0]
            # 可疑度图也要存:过杀率要求良品走**同一条**选窗链路,
            # 否则良品与缺陷品的处理不对称,过杀率没有意义
            g = np.asarray(Image.open(ip).convert('L'))
            sus = patch_suspicion(
                suspicion_map(normalize_illumination(g, cv), cv), GRID)
            np.savez(f, full=full[0], w3=w3, w5=w5, susp=sus)
            n_new += 1
        n += 1
        print(f"  [{cls}] good {n} (新增 {n_new})", flush=True)
    (d / 'done').touch()
    print(f"[done] {cls} ({n} 张良品,本次新增 {n_new})", flush=True)
