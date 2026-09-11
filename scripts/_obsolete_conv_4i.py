"""⚠️ 本脚本已作废(2026-09-11),不要使用。

项目里已有**权威** 4i 转换:`scripts/make_4i_mvtec.py` → `data/surface_defects_4i/`
(远程 `/root/autodl-tmp/surface_defects_4i`,已出过发表数字 93.7→94.8 / 68.9→77.1)。

本脚本是另写的一份,**与权威版不一致**:

    test/good   权威 min(50, N//2)  → 50     本脚本 精确对半 → 100
    GT          权威 转换期 >0 落 {0,255}     本脚本 原样拷贝,下游 >128
    全零 GT     权威 已剔除(MT_Break_8)      本脚本 未剔除
    缺陷目录名  权威 abrasion_mask/patches   本脚本 abrasion_mark/patch
    物体名      权威 4i_prompt_map.json       本脚本 手写 classes4i.OBJECT

产物 `data/4i_mvtec/` 已重命名为 `data/_OBSOLETE_4i_mvtec_不要用/`。

保留本文件仅为留痕。新增同类需求请直接用 make_4i_mvtec.py。

---------------- 以下是原文,已被上面的说明取代 ----------------
Surface Defects-4i → MVTec-AD 目录布局(让本地/远程同一套链路直接吃)。

    源(TGRNet/MSD-Seg2)              目标(data/4i_mvtec/<cls>/)
      Images/*.png  (缺陷, 200²)  →   test/<defect>/*.png
      GT/*.png      (掩膜, 同名)   →   ground_truth/<defect>/*_mask.png
      Nd/*.jpg      (正常)         →   train/good/*.png   (前 --train-frac)
                                       test/good/*.png    (其余)

## 几个必须显式处理的坑

1. **GT 格式类间不一致**:7 类是 0/255 二值,5 类(Leather/MT_*/Tile)是
   256 级灰阶。本脚本**原样搬运**,不做二值化 —— 判定口径由加载器统一
   `resize(240,BILINEAR) > 128` 决定(与 MVTec 实验同一行代码)。
   灰阶那 5 类在 >0 与 >128 之间差 1.7~3.1pt 面积(边界环),换规则会
   改变像素指标,因此**规则必须写死在报告里**,不能两边各用一套。

2. **Nd 池是跨类共享的**(7 类共用同一个 200 张池,md5 逐文件一致)。
   本脚本对每个类**各切一次**,用固定的 `--train-frac` 与固定种子。
   因此这 7 类会得到**同一批** train/good 与 test/good —— 这是数据集属性,
   不是 bug;分析时(见 shot_scaling.py)还要在 train/good 内部再切
   gallery 池 / 评估良品。

3. **分辨率/模式**:缺陷图 200×200、Nd 有 224×224 也有 200×200,全部灰度。
   统一转 PNG 灰度;放大到 240 由 `engine.preprocess_rgb` 负责。

4. **文件名唯一性**:不同类的缺陷图可能重名,故目标扁平到各自类目录下,
   缺陷名进文件名前缀以防同类内重名(4i 里 Images/GT 同名且类内唯一)。

用法:
    python scripts/conv_4i.py --src /home/asus/桌面/JD/TGRNet/MSD-Seg2 \
        --dst data/4i_mvtec --train-frac 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from classes4i import CLASSES_4I, CLS2SRC, DEFECT_KEY          # noqa: E402


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/asus/桌面/JD/TGRNet/MSD-Seg2")
    ap.add_argument("--dst", default="data/_OBSOLETE_4i_mvtec_不要用")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="Nd 池中划给 train/good 的比例(其余给 test/good)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        print(f"[FAIL] 源目录不存在: {src}")
        return 1

    # ---- 先核对共享池假设是否仍成立(md5),否则设计前提变了 ----
    pools = {}
    for cls in CLASSES_4I:
        nd = src / CLS2SRC[cls] / "Nd"
        pools[cls] = hashlib.md5(
            "".join(sorted(f"{f.name}:{md5(f)}" for f in nd.glob("*"))
                    ).encode()).hexdigest()
    by_pool = {}
    for c, h in pools.items():
        by_pool.setdefault(h, []).append(c)
    print("Nd 池分组(md5 相同 = 同一批图):")
    for h, cs in by_pool.items():
        print(f"  {h[:12]}  {len(cs)} 类: {', '.join(cs)}")
    print()

    rng = np.random.default_rng(args.seed)
    total = {"defect": 0, "train": 0, "test_good": 0}
    print(f"{'类':13s} {'源目录':11s} {'缺陷':>5s} {'Nd':>5s} "
          f"{'train':>6s} {'test_good':>10s}  GT模式")
    for cls in CLASSES_4I:
        s = src / CLS2SRC[cls]
        d = dst / cls
        defect = DEFECT_KEY[cls]
        (d / "train" / "good").mkdir(parents=True, exist_ok=True)
        (d / "test" / "good").mkdir(parents=True, exist_ok=True)
        (d / "test" / defect).mkdir(parents=True, exist_ok=True)
        (d / "ground_truth" / defect).mkdir(parents=True, exist_ok=True)

        # --- 缺陷图 + 掩膜 ---
        imgs = sorted((s / "Images").glob("*"))
        n_def = 0
        gts = {f.stem: f for f in (s / "GT").glob("*")}
        for ip in imgs:
            op = d / "test" / defect / f"{cls}__{ip.stem}.png"
            Image.open(ip).convert("L").save(op)
            gp = gts.get(ip.stem)
            if gp is not None:
                Image.open(gp).convert("L").save(
                    d / "ground_truth" / defect / f"{op.stem}_mask.png")
            n_def += 1

        # --- 正常池 → train/good + test/good ---
        nds = sorted((s / "Nd").glob("*"))
        perm = rng.permutation(len(nds))
        n_tr = int(round(len(nds) * args.train_frac))
        tr_i, te_i = perm[:n_tr], perm[n_tr:]
        for j, i in enumerate(tr_i):
            Image.open(nds[i]).convert("L").save(
                d / "train" / "good" / f"{cls}_nd{j:04d}.png")
        for j, i in enumerate(te_i):
            Image.open(nds[i]).convert("L").save(
                d / "test" / "good" / f"{cls}_nd{j:04d}.png")

        # GT 模式(只抽首图看)
        g0 = np.asarray(Image.open(sorted((s / "GT").glob("*"))[0]))
        mode = "二值" if len(np.unique(g0)) <= 2 else f"灰阶({len(np.unique(g0))})"

        total["defect"] += n_def
        total["train"] += n_tr
        total["test_good"] += len(nds) - n_tr
        print(f"{cls:13s} {CLS2SRC[cls]:11s} {n_def:5d} {len(nds):5d} "
              f"{n_tr:6d} {len(nds)-n_tr:10d}  {mode}")

    print(f"\n合计: 缺陷 {total['defect']} | train/good {total['train']} | "
          f"test/good {total['test_good']}")
    print(f"输出: {dst.resolve()}")
    print("\n下一步(本地 OpenVINO CPU,无需 GPU):")
    print(f"  python scripts/build_text_protos.py --classes "
          f"<i4_*> --out data/deploy/text_protos_4i")
    print(f"  python scripts/cache_good.py --data_root {dst} --deploy data/deploy "
          f"--device cpu --classes <i4_*> --out /tmp/i4_cache_good")
    print(f"  python scripts/line_experiment.py --data_root {dst} "
          f"--deploy data/deploy --device cpu --text data/deploy/text_protos_4i "
          f"--classes <i4_*> --n 999 --shots 4 --cache /tmp/i4_cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
