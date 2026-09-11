"""校验 make_4i_mvtec.py 的产物是否满足 evaluate.py 的全部隐含假设。

**为什么需要这一步**:evaluate.py 读 mask 是
    np.asarray(Image.open(m).convert("L").resize((240,240), BILINEAR)) > 128
如果某类 mask 在 200→240 双线性 + `>128` 之后**整幅变空**,该类像素 AUROC 会静默
失真(nan 或退化),而日志里只会显示一个看着正常的百分数。所以这里**逐张复刻**
这条读法,确认没有一张图被吃掉。

用法:
    python scripts/verify_4i_layout.py --root data/surface_defects_4i
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mvtec_load_mask(p: Path) -> np.ndarray:
    """逐字复刻 evaluate.py 的 mask 读法。"""
    return np.asarray(Image.open(p).convert("L")
                      .resize((240, 240), Image.BILINEAR)) > 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"{'类':10s} {'缺陷':>4s} {'train/good':>10s} {'test/good':>9s} "
          f"{'交叠':>4s} {'空mask':>6s} {'>128前景占比':>12s}")
    print("-" * 70)
    bad = []
    for c in classes:
        d = root / c
        gt_dir = d / "ground_truth"
        tr = sorted(p.stem for p in (d / "train" / "good").glob("*.png"))
        te = sorted(p.stem for p in (d / "test" / "good").glob("*.png"))
        overlap = set(tr) & set(te)

        masks = sorted(gt_dir.rglob("*_mask.png"))
        empty, fgs = 0, []
        for m in masks:
            b = mvtec_load_mask(m)
            fg = float(b.mean())
            fgs.append(fg)
            if fg == 0.0:
                empty += 1

        # 每张缺陷图都要有同名 mask
        n_def = sum(1 for p in (d / "test").iterdir()
                    if p.is_dir() and p.name != "good"
                    for _ in p.glob("*.png"))
        miss = n_def - len(masks)
        if overlap or empty or miss:
            bad.append(f"{c}: 交叠{len(overlap)} 空mask{empty} 缺mask{miss}")

        print(f"{c:10s} {n_def:4d} {len(tr):10d} {len(te):9d} "
              f"{len(overlap):4d} {empty:6d} "
              f"{np.mean(fgs)*100:11.2f}%")

    print("-" * 70)
    if bad:
        print("[FAIL] " + " | ".join(bad))
        return 1
    print("[PASS] 结构完整:train/test 良品互斥、每张缺陷图都有非空 mask(240px>128 后仍非空)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
