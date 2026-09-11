"""生产像素链的**本地零推理复现** —— 像素级修正实验的地基。

## 为什么能做

`pipeline.py:178-207` 的图是**可分离**的:

    m_all = harmonic_3way(cls_prob, m48_zero, m32_zero)   ← 与 gallery 无关的常数项
          + few_map(gallery)                              ← 只有这项随入库变化

而 zero 项的三样原料全部已在 `/tmp/feat_cache{,_good}` 里:

    cls_prob = _prob(full[0:1], pos, neg, temp)            full[0] 就是 CLS
    m48      = _scatter_harmonic(_prob(w3, ...), idx3)     w3 就是 169 个 3×3 窗口 CLS
    m32      = _scatter_harmonic(_prob(w5, ...), idx2)     w5 就是 196 个 2×2 窗口 CLS

`cache_good.py:102-105` 证实 `w3/w5 = eng.tower_w(toks[0][seq])[:, 0]`,与
`_window_feats` 返回的正是同一个东西。文本原型 `data/deploy/text_protos/<cls>.npz`
含 normal/abnormal/temp(=100.0),15 类齐全。

⇒ **整条生产像素链可以本地重放,零次推理、不用远程。**

## 本脚本干什么

按 evaluate.py 的口径重放 zero-shot,和归档数字对表:

    归档(run_final_v3, 15 类, seed42):zero-shot image 90.1 / pixel 80.8

**打中 ⇒ 复现忠实,可以在此之上做像素级修正实验;打不中 ⇒ 先定位差异,不许调参凑数。**

## 口径(必须与 evaluate.py 逐条一致)

    map 15×15 --F.interpolate(bilinear, align_corners=False)--> 240×240
    GT       --Image.BILINEAR resize 到 240 --> >128 --> bool   (缓存里已按此存好)
    image    roc_auc_score(良/缺标签, cls_prob)
    pixel    roc_auc_score(全部图全部像素拼起来, map; GT 同序拼起来)

用法:
    python scripts/exp/pixel_replay.py                 # 15 类全跑
    python scripts/exp/pixel_replay.py bottle,screw    # 指定类
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402

from runtime.pipeline import GRID, N_PATCH, OVPipeline   # noqa: E402

ALL15 = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="all")
    ap.add_argument("--good", default="/tmp/feat_cache_good")
    ap.add_argument("--bad", default="/tmp/feat_cache")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--deploy", default="data/deploy")
    return ap.parse_args()


def load_npz(d, only_gt=False):
    """按文件名排序读 npz。only_gt=True 时只保留 gt 非空的(真缺陷图)。"""
    out = []
    for f in sorted(Path(d).glob("*.npz")):
        if f.name == "gallery.npz":
            continue
        z = np.load(f)
        if only_gt:
            if "gt" not in z.files or not z["gt"].any():
                continue
        out.append(z)
    return out


def up_to_gt(m225: np.ndarray, size=(240, 240)) -> np.ndarray:
    """(225,) → (240,240) 展平。与 evaluate.py::up_to_gt 同源。"""
    g = torch.from_numpy(m225.reshape(1, 1, GRID, GRID).astype(np.float32))
    return F.interpolate(g, size=size, mode="bilinear",
                         align_corners=False).flatten().numpy()


def zero_maps(z, pos, neg, temp, idx3, idx2):
    """缓存 npz → (m_all (225,), cls_prob 标量)。与 pipeline.py:154-185 同源。"""
    cls_prob = float(OVPipeline._prob(z["full"][:1], pos, neg, temp)[0])
    inv = np.full(N_PATCH, 1.0 / cls_prob, dtype=np.float32)
    n_terms = np.ones(N_PATCH, dtype=np.float32)          # CLS 恒在
    for key, idx in (("w3", idx3), ("w5", idx2)):
        wp = OVPipeline._prob(z[key], pos, neg, temp)
        m, cnt = OVPipeline._scatter_harmonic(wp, idx)
        present = cnt > 0
        inv[present] += 1.0 / m[present]
        n_terms[present] += 1.0
    return (n_terms / inv).astype(np.float32), cls_prob


def main() -> int:
    a = parse_args()
    classes = ALL15 if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]
    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")

    print("生产像素链本地零推理复现 | 口径 = evaluate.py (map bilinear→240, GT >128)")
    print(f"  文本原型 {a.text}   win_idx_k3{idx3.shape} win_idx_k2{idx2.shape}")
    print("  靶子:归档 zero-shot 15 类 image 90.1 / pixel 80.8 (run_final_v3, seed42)")
    print()
    print(f"{'类':13s} {'良品':>4s} {'缺陷':>4s} {'image':>7s} {'pixel':>7s}")
    print("-" * 46)

    rows = []
    for cls in classes:
        tp = Path(a.text) / f"{cls}.npz"
        if not tp.exists():
            print(f"{cls:13s} 文本原型缺失,跳过")
            continue
        d = np.load(tp)
        pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])

        good = load_npz(Path(a.good) / cls)
        bad = load_npz(Path(a.bad) / cls, only_gt=True)
        if not good or not bad:
            print(f"{cls:13s} 缓存不全,跳过")
            continue

        s_good = [zero_maps(z, pos, neg, temp, idx3, idx2)[1] for z in good]
        out_bad = [zero_maps(z, pos, neg, temp, idx3, idx2) for z in bad]
        s_bad = [cp for _m, cp in out_bad]

        img_auc = roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(bad))],
                                np.r_[s_good, s_bad]) * 100
        px_s = np.concatenate([up_to_gt(m) for m, _cp in out_bad])
        px_g = np.concatenate([z["gt"].flatten() for z in bad])
        px_auc = roc_auc_score(px_g, px_s) * 100

        rows.append((cls, img_auc, px_auc, len(good), len(bad)))
        print(f"{cls:13s} {len(good):4d} {len(bad):4d} "
              f"{img_auc:7.1f} {px_auc:7.1f}")

    print("-" * 46)
    if rows:
        im = np.mean([r[1] for r in rows])
        pm = np.mean([r[2] for r in rows])
        print(f"{'15类均值':13s} {'':4s} {'':4s} {im:7.1f} {pm:7.1f}")
        print(f"{'归档靶子':13s} {'':4s} {'':4s} {90.1:7.1f} {80.8:7.1f}")
        print(f"{'差':13s} {'':4s} {'':4s} {im-90.1:+7.1f} {pm-80.8:+7.1f}")
        print(f"\n  类数={len(rows)}")
        print("  ★ 差在 ±0.5pt 内且无系统性偏移 ⇒ 复现忠实;")
        print("     否则先定位差异来源(文本原型?窗口索引?GT 口径?),"
              "不许调参把数字凑上去。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
