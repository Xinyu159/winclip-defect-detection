"""Surface Defects-4i(MSD-Seg2)→ MVTec-AD 目录布局,供 evaluate.py 直接吃。

    <out>/<Class>/train/good/*.png             # Nd 画廊(few-shot gallery 池)
    <out>/<Class>/test/good/*.png              # 留出的 Nd(负样本,只影响 img AUROC)
    <out>/<Class>/test/<defect>/*.png          # Images 缺陷图
    <out>/<Class>/ground_truth/<defect>/<stem>_mask.png

## 三个必须解释的口径决定

1) **train/test 良品互斥(与手册字面写法不同,是手册的漏洞)**
   手册写「train/good=Nd、test/good=Nd 抽 50」——两者都取自同一个 Nd 池,**有交叠**。
   evaluate.py 的 `make_gallery` 从 `train/good` 采 shot 张当 gallery,若 test/good 与
   其交叠,则 4-shot 下每张测试良品约 4/N 的概率**已经在 gallery 里**,会把良品分数
   抬高 → 虚高 AUROC。故此处**先把 Nd 切成互斥两半**:test/good 抽 min(50, N//2),
   其余全部进 train/good。抽签固定 seed=42,可复现。

2) **GT 统一预二值化为 {0,255}(用 `>0`)—— 顺带消掉一处跨项目口径冲突**
   `evaluate.py` 读 mask 时是 `resize(240, BILINEAR) > 128`;而 `C++传统版4i升级手册.md`
   §2 写的是「像素真值由 GT>0 得」。对 7 个硬标签类(0/255)两者等价,**但对 5 个软标签类
   (Leather/MT_Break/MT_Fray/MT_Uneven/Tile,GT 是 0~255 连续灰阶,像高斯糊过的边界)
   差别很大**。这里在转换期就按 `>0` 落成 {0,255},于是下游的 `>128` 退化为恒等 ——
   **两个项目因此使用同一个真值定义**,README 对照表才成立。软标签的「光晕」占比在
   清单 json 里逐类记录,供敏感性讨论。

3) **统一到 200×200**:Nd 多为 224×224(7/12 类),Images/GT 皆为 200×200。
   全图统一 200×200(LANCZOS)。理由:200 是缺陷图原生尺寸,且 C++ 手册 §3.A.1 要求
   「所有图处理到同尺寸 200×200」,不统一则两边不是同一套输入。

用法:
    python scripts/make_4i_mvtec.py --src ~/桌面/JD/TGRNet/MSD-Seg2 \
        --out data/surface_defects_4i --manifest logs/4i_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# 输出类名 → (源目录名, CPE 物体名, 缺陷类型目录名)
# 物体名取自论文正文的材料名([39] = MVTec AD 论文,故 Leather/Tile 与 MVTec 同源);
# 缺陷类型取自论文对各类的英文描述。
#
# ⚠ 源目录名与输出名对 Al_Rm **故意不一致**:数据集自己的目录叫 `AI_Rm`(A-I),
#   而兄弟目录叫 `Al_Con`(A-l),论文正文两处都写 "aluminum surface" —— 是数据集
#   的拼写不一致。输出统一成 `Al_Rm`,源名留在 manifest 里以便回溯。
#   (`AI_Rm` 与 `Al_Rm` 在多数等宽字体下几乎同形,直接照抄会把这个坑传下去)
CLASSES = {
    "Al_Rm":     ("AI_Rm",          "aluminum surface", "rub_mark"),
    "Al_Con":    ("Al_Con",         "aluminum surface", "convexity"),
    "MT_Uneven": ("MT_Uneven",      "magnetic tile",    "uneven"),
    "MT_Break":  ("MT_Break",       "magnetic tile",    "break"),
    "MT_Fray":   ("MT_Fray",        "magnetic tile",    "fray"),
    "Steel_Ld":  ("Steel_Ld",       "steel surface",    "liquid"),
    "Steel_Am":  ("Steel_Am",       "steel surface",    "abrasion_mask"),
    "Steel_Pa":  ("Steel_Pa",       "steel surface",    "patches"),
    "Steel_Sc":  ("Steel_Sc",       "steel surface",    "scratches"),
    # 以下三类论文只说「leather 与 tile 是非金属数据」,未给缺陷子类名 → 统一 defect
    "Rail":      ("Rail",           "rail surface",     "defect"),
    "Leather":   ("Leather",        "leather",          "defect"),
    "Tile":      ("Tile",           "tile",             "defect"),
}

SIZE = 200
SEED = 42


def load_l(p: Path) -> Image.Image:
    return Image.open(p).convert("L")


def save_l(im: Image.Image, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, format="PNG", optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MSD-Seg2 根目录")
    ap.add_argument("--out", required=True, help="输出 MVTec 布局根目录")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--prompt-map-out", default="",
                    help="写 {目录名: CPE 物体名} 给 evaluate.py --prompt-map。"
                         "由本脚本生成而非手写,保证与目录名不会漂移。")
    ap.add_argument("--test-good-max", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest = {"src": str(src), "size": SIZE, "seed": args.seed,
                "test_good_max": args.test_good_max,
                "gt_rule": "raw >0 落 {0,255};下游 >128 因此恒等",
                "split_rule": "test/good 与 train/good 互斥(防 gallery 泄漏)",
                "classes": {}}

    for cls, (src_name, obj, dtype) in CLASSES.items():
        cdir = src / src_name
        if not cdir.is_dir():
            print(f"[FATAL] 源目录不存在: {cdir}")
            return 1

        imgs = sorted((cdir / "Images").glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
        nds = sorted((cdir / "Nd").glob("*"))
        nds = [p for p in nds if p.suffix.lower() in (".jpg", ".png", ".jpeg")]

        # --- Nd 互斥切分 ---
        n_test = min(args.test_good_max, len(nds) // 2)
        perm = np.random.default_rng(args.seed).permutation(len(nds))
        te_idx = set(int(i) for i in perm[:n_test])

        nd_sizes, nd_test_n, nd_train_n = {}, 0, 0
        for i, p in enumerate(nds):
            im = load_l(p)
            nd_sizes[im.size] = nd_sizes.get(im.size, 0) + 1
            stem = p.stem
            if i in te_idx:
                save_l(im.resize((SIZE, SIZE), Image.LANCZOS),
                       out / cls / "test" / "good" / f"{stem}.png")
                nd_test_n += 1
            else:
                save_l(im.resize((SIZE, SIZE), Image.LANCZOS),
                       out / cls / "train" / "good" / f"{stem}.png")
                nd_train_n += 1

        # --- 缺陷图 + GT ---
        fg_raw, halo, missing, dropped = [], [], [], []
        for p in imgs:
            stem = p.stem
            gt = cdir / "GT" / f"{stem}.png"
            if not gt.exists():
                missing.append(stem)
                continue
            # 先算 mask 再决定收不收:源数据里存在**整幅全零的 GT**(无标注),
            # 这种样本评不了像素级,且它的高分像素会被当成假阳**压低**该类 AUROC。
            g = np.asarray(load_l(gt).resize((SIZE, SIZE), Image.NEAREST))
            binm = (g > 0).astype(np.uint8) * 255
            if int(binm.max()) == 0:
                dropped.append(stem)
                continue

            save_l(load_l(p).resize((SIZE, SIZE), Image.LANCZOS),
                   out / cls / "test" / dtype / f"{stem}.png")
            fg_raw.append(float((g > 0).mean()))
            halo.append(float(((g > 0) & (g <= 128)).mean()))
            save_l(Image.fromarray(binm, "L"),
                   out / cls / "ground_truth" / dtype / f"{stem}_mask.png")

        manifest["classes"][cls] = {
            "src_dir": src_name, "object": obj, "defect_type": dtype,
            "n_defect": len(fg_raw), "n_train_good": nd_train_n,
            "n_test_good": nd_test_n,
            "nd_native_sizes": {f"{w}x{h}": c for (w, h), c in nd_sizes.items()},
            "gt_fg_ratio_mean": round(float(np.mean(fg_raw)), 4) if fg_raw else None,
            "gt_fg_ratio_min": round(float(np.min(fg_raw)), 4) if fg_raw else None,
            "gt_fg_ratio_max": round(float(np.max(fg_raw)), 4) if fg_raw else None,
            "gt_halo_frac_of_fg": (round(float(np.mean(halo) / max(np.mean(fg_raw), 1e-9)), 3)
                                   if fg_raw else None),
            "missing_gt": missing,
            "dropped_empty_gt": dropped,   # 源数据全零 GT,已剔除(见上)
        }
        print(f"[{cls:10s}] obj={obj:16s} defect={dtype:14s} "
              f"缺陷 {len(fg_raw):3d} | train/good {nd_train_n:3d} "
              f"| test/good {nd_test_n:3d} | Nd原生 {nd_sizes}"
              + (f"  ⚠缺GT {missing}" if missing else "")
              + (f"  ⚠剔除全零GT {dropped}" if dropped else ""))

    # --- 自检 ---
    bad = []
    for cls in manifest["classes"]:
        d = out / cls
        n_tr = len(list((d / "train" / "good").glob("*.png")))
        n_te = len(list((d / "test" / "good").glob("*.png")))
        if n_tr == 0 or n_te == 0:
            bad.append(f"{cls}: train={n_tr} test={n_te}")
    if bad:
        print("[FATAL] 空目录:" + "; ".join(bad))
        return 1

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[manifest] {args.manifest}")

    if args.prompt_map_out:
        pm = {c: v["object"] for c, v in manifest["classes"].items()}
        Path(args.prompt_map_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prompt_map_out).write_text(
            json.dumps(pm, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[prompt-map] {args.prompt_map_out}  ({len(set(pm.values()))} 个物体名"
              f" / {len(pm)} 个目录)")

    tot = sum(c["n_defect"] + c["n_train_good"] + c["n_test_good"]
              for c in manifest["classes"].values())
    print(f"[done] {len(manifest['classes'])} 类 / {tot} 张 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
