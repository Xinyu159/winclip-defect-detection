"""数据集冻结清单 —— 把"数据集的事实"从记忆搬进仓库。

## 为什么有这个脚本

反复因为数据集出错的根因不是"看错了",是**事实散落在记忆和各脚本里,没有单一权威**。
已经踩过的:4i 的 train/test 良品池不互斥(虚高 AUROC)、源数据全零 GT 被当成正常样本、
224px 混 200px、类名 `AI_Rm`/`Al_Con` 拼写不一致、train/good 张数记成 3529(实数 3629)。

⇒ 本脚本一次生成清单,**以后任何"我以为有 N 张"当场被校验挡住**。

## 冻结什么

| 项 | 抓什么错 |
|---|---|
| 每类每划分的**文件数** + **文件名列表哈希** | 增删改名、顺序变化 |
| **两套独立枚举互拍**(pathlib 裸扫 vs `mvtec.py` 的加载器) | 连加载器的 bug 也一起抓 |
| train/good 与 test/good 的**内容哈希交集** | 良品池泄漏(few-shot 自引用的根) |
| 每张缺陷图**有没有对应 GT**;GT 全零的**逐个列名** | 4i 那次的坑 |
| 图像尺寸 / GT 尺寸分布 | 224 混 200 那次的坑 |
| 特征缓存与数据集**逐类对账** | 缓存少一张、多一张、错配 |

## 两套枚举为什么必须独立

若两边都调 `mvtec.iter_test_images`,那只是"加载器跟自己对拍",永远相等。
所以一边是 `pathlib.glob` 裸扫(只认 MVTec 的目录约定),一边是加载器。
两边不一致 ⇒ 加载器有 bug,不是数据有问题 —— **报出来,别抹平**。

用法:
    python scripts/make_dataset_manifest.py            # 生成 data/dataset_manifest.json
    python scripts/make_dataset_manifest.py --check    # 与已冻结的清单对拍,漂移则退出码 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402

import mvtec                                             # noqa: E402

MANIFEST = REPO_ROOT / "data" / "dataset_manifest.json"

# P2 后台作业覆盖的类 —— 两条命令的并集,逐字抄自 `ps`(2026-09-12)。
# 第一条 `p2_a.log`: hazelnut,carpet,wood,zipper,cable,capsule,bottle   (7 类)
# 第二条 `p2_c.log`: metal_nut,pill,screw,tile,toothbrush,transistor,grid,leather (8 类)
# 为什么写死:我一度把 P2 记成"15 类 3629 张",实际第一条只列了 7 类 1810 张。
# 第二条是用户 2026-09-12 拍板补的 —— **在补起来之前,§五 判据(≥8/15 类为正)算不出判决**。
P2_CLASSES = {"hazelnut", "carpet", "wood", "zipper", "cable", "capsule", "bottle",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
              "grid", "leather"}


def sha1_list(items) -> str:
    """文件名列表的哈希 —— 增删改名都会变,与顺序绑定。"""
    h = hashlib.sha1()
    for s in items:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def sha1_file(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def img_size_hw(p: Path):
    with Image.open(p) as im:
        return f"{im.size[0]}x{im.size[1]}"


def scan_class(root: Path, cls: str) -> dict:
    """一个类的裸扫(pathlib,只认 MVTec 目录约定)+ 加载器对拍。"""
    cdir = root / cls
    out: dict = {"defects": {}, "warnings": []}

    # ---- 裸扫:train/good ----
    tg = sorted((cdir / "train" / "good").glob("*.png"))
    out["train_good"] = {"n": len(tg), "list_sha1": sha1_list([p.name for p in tg]),
                         "names": [p.name for p in tg]}

    # ---- 裸扫:test/<子目录> ----
    test_dir = cdir / "test"
    subs = sorted(p for p in test_dir.iterdir() if p.is_dir()) if test_dir.is_dir() else []
    tg_names = sorted((test_dir / "good").glob("*.png")) if (test_dir / "good").is_dir() else []
    out["test_good"] = {"n": len(tg_names),
                        "list_sha1": sha1_list([p.name for p in tg_names]),
                        "names": [p.name for p in tg_names]}

    n_gt_missing, gt_allzero, sizes, gt_sizes = 0, [], {}, {}
    for sub in subs:
        if sub.name == "good":
            continue
        imgs = sorted(sub.glob("*.png"))
        names, nz = [], []
        for p in imgs:
            names.append(p.name)
            sizes[img_size_hw(p)] = sizes.get(img_size_hw(p), 0) + 1
            cand = cdir / "ground_truth" / sub.name / f"{p.stem}_mask.png"
            if not cand.exists():
                n_gt_missing += 1
                continue
            gt_sizes[img_size_hw(cand)] = gt_sizes.get(img_size_hw(cand), 0) + 1
            arr = np.asarray(Image.open(cand).convert("L"))
            if arr.max() == 0:
                gt_allzero.append(f"{sub.name}/{p.name}")
            else:
                nz.append(p.name)
        out["defects"][sub.name] = {
            "n": len(imgs), "list_sha1": sha1_list(names),
            "n_gt_nonzero": len(nz), "names": names}

    for p in tg_names:
        sizes[img_size_hw(p)] = sizes.get(img_size_hw(p), 0) + 1
    out["_sizes"] = sizes
    out["_gt_sizes"] = gt_sizes
    out["n_gt_missing"] = n_gt_missing
    out["gt_allzero"] = sorted(gt_allzero)

    # ---- 加载器对拍(独立第二套枚举)----
    ldr = list(mvtec.iter_test_images(root, cls))
    ldr_good = [n for n, rel, _p, _m in ldr if rel == "good"]
    ldr_def = [n for n, rel, _p, _m in ldr if rel != "good"]
    n_bare_def = sum(v["n"] for v in out["defects"].values())
    out["_crosscheck"] = {
        "train_good": len(mvtec.iter_train_images(root, cls)) == len(tg),
        "test_good": sorted(ldr_good) == sorted(p.name for p in tg_names),
        "defect_n": len(ldr_def) == n_bare_def,
        "n_loader": len(ldr),
    }
    for k, ok in out["_crosscheck"].items():
        if k != "n_loader" and not ok:
            out["warnings"].append(f"★ 两套枚举不一致:{k}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--good-cache", default="/tmp/feat_cache_good")
    ap.add_argument("--bad-cache", default="/tmp/feat_cache")
    ap.add_argument("--train-cache", default="/tmp/feat_cache_train_good")
    ap.add_argument("--check", action="store_true",
                    help="与已冻结清单对拍;有漂移则退出码 1")
    a = ap.parse_args()

    root = Path(a.root)
    classes = [c for c in mvtec.MVTEC_CLASSES if (root / c).is_dir()]
    if not classes:
        print(f"★ {root} 下没有 MVTec 类目录")
        return 2

    man: dict = {"root": str(a.root), "classes": {}}
    warns: list[str] = []
    for cls in classes:
        man["classes"][cls] = scan_class(root, cls)

    # ---- 良品池互斥:train/good ∩ test/good 的文件名 + 内容哈希 ----
    print("=" * 92)
    print("MVTec 15 类冻结清单")
    print("=" * 92)
    print(f"{'类':12s}{'train/good':>11s}{'test/good':>10s}{'缺陷':>6s}"
          f"{'GT全零':>7s}{'缺GT':>6s}{'尺寸':>14s}{'互斥':>7s}")
    tot = [0, 0, 0]
    for cls in classes:
        c = man["classes"][cls]
        tg, eg = c["train_good"]["n"], c["test_good"]["n"]
        nd = sum(v["n"] for v in c["defects"].values())
        tot[0] += tg
        tot[1] += eg
        tot[2] += nd
        ov = set(c["train_good"]["names"]) & set(c["test_good"]["names"])
        sz = ", ".join(f"{k}×{v}" for k, v in sorted(c["_sizes"].items()))
        # ⚠️ 文件名撞车**不是**泄漏:MVTec 的 train/good 与 test/good 各自从 000.png 起编号。
        # 我第一版把重名报了 15 条警告 —— 那是判据写错了。唯一有效的判据是内容哈希(下一节)。
        print(f"{cls:12s}{tg:11d}{eg:10d}{nd:6d}{len(c['gt_allzero']):7d}"
              f"{c['n_gt_missing']:6d}{sz:>14s}{'同号' if ov else '—':>7s}")
        man["classes"][cls]["_overlap_names"] = sorted(ov)
        warns.extend(f"{cls}: {w}" for w in c["warnings"])
        if c["gt_allzero"]:
            warns.append(f"{cls}: {len(c['gt_allzero'])} 张 GT 全零 "
                         f"(会被当正常样本): {c['gt_allzero'][:3]}")

    print("-" * 92)
    print(f"{'合计':12s}{tot[0]:11d}{tot[1]:10d}{tot[2]:6d}")

    # ---- 内容哈希互斥(重名检查只是第一道)----
    print("\n良品池内容哈希互斥(train/good vs test/good):")
    leak = []
    for cls in classes:
        c = man["classes"][cls]
        ht = {sha1_file(root / cls / "train" / "good" / n)
              for n in c["train_good"]["names"]}
        he = {sha1_file(root / cls / "test" / "good" / n)
              for n in c["test_good"]["names"]}
        both = ht & he
        if both:
            leak.append((cls, len(both)))
        print(f"  {cls:12s} train {len(ht):4d} 张 / test {len(he):4d} 张  "
              f"交集 {len(both)}  {'✓' if not both else '✗ 内容重复!'}")
    if leak:
        warns.append(f"★ 良品池内容重复:{leak}")

    # ---- 特征缓存对账 ----
    #
    # ★ 判据是**张数对不对**,`done` 只是旁证。
    # `done` 由 cache_good.py 写;缺陷缓存与 test/good 是另外的脚本建的,从不写。
    # 所以「没有 done」不能当"没建完",而「张数不足」永远是真的没建完。
    # 坑在此:进程被杀在半路 ⇒ 目录里有几十张、张数不足、无 done。数一眼"有 9 张"
    # 就以为建过 —— 2026-09-12 抓到的 screw 正是这样。
    def state(d: Path, expect: int):
        """→ (状态串, 张数, 是否可用)"""
        if not d.is_dir():
            return "○ 未建", 0, False
        n = len(list(d.glob("[0-9]*.npz")))
        done = (d / "done").exists()
        if n == expect:
            return ("✓ 完成" if done else "✓ 完成ᵈ", n, True)
        if n == 0:
            return "○ 未建", 0, False
        if n > expect:
            return f"✗ 超量 {n}>{expect}", n, False
        return f"⏳ {n}/{expect}", n, False

    print("\n特征缓存对账(判据 = 张数;ᵈ = 有 done 标记):")
    print(f"  {'类':12s}{'test/good':>16s}{'缺陷':>16s}{'train/good':>18s}")
    cache = {}
    for cls in classes:
        c = man["classes"][cls]
        n_eg = c["test_good"]["n"]
        n_df = sum(v["n"] for v in c["defects"].values())
        n_tg = c["train_good"]["n"]

        sg, ng, ok_g = state(Path(a.good_cache, cls), n_eg)
        sb, nb, ok_b = state(Path(a.bad_cache, cls), n_df)
        st, nt, ok_t = state(Path(a.train_cache, cls), n_tg)
        cache[cls] = {"test_good": ng, "defect": nb, "train_good": nt,
                      "test_good_ok": ok_g, "defect_ok": ok_b,
                      "train_good_ok": ok_t}

        print(f"  {cls:12s}{sg:>16s}{sb:>16s}{st:>18s}")
        for s, lbl in ((sg, "test/good"), (sb, "缺陷"), (st, "train/good")):
            if s.startswith("✗"):
                warns.append(f"{cls}: {lbl} 缓存张数不符 —— {s}")
        # 半程且没有任何进程会来补完它 = 残留,将来会被误读成"建过一部分"
        if st.startswith("⏳") and cls not in P2_CLASSES:
            warns.append(f"{cls}: train/good 残留 {st.split()[1]} —— **不在 P2 的类别列表"
                         f"里**,没有进程会补全它。别把这 {nt} 张当成'建过一部分'")
    man["feature_caches"] = cache
    man["p2_classes"] = sorted(P2_CLASSES)

    n_ok = sum(1 for c in classes if cache[c]["train_good_ok"])
    n_have = sum(cache[c]["train_good"] for c in classes)
    n_exp = sum(man["classes"][c]["train_good"]["n"] for c in classes)
    n_p2 = sum(man["classes"][c]["train_good"]["n"] for c in classes
               if c in P2_CLASSES)
    print(f"\n  train/good 张数齐全的类 {n_ok}/{len(classes)},合计 {n_have}/{n_exp} 张")
    print(f"  ★ P2 后台作业只覆盖 {len(P2_CLASSES)}/{len(classes)} 类(共 {n_p2} 张);"
          f"另外 {len(classes) - len(P2_CLASSES)} 类**没有任何进程在跑**")
    if n_ok < len(classes):
        orphan = sorted(c for c in classes
                        if not cache[c]["train_good_ok"] and c not in P2_CLASSES)
        if orphan:
            # 这才是真问题:有半程的残留,却没有任何在跑的作业会补完它
            warns.append(f"train/good 有半程残留但无作业覆盖:{orphan} —— "
                         f"别把这几十张当成'建过一部分'")
        else:
            print(f"  (train/good 仍在建:{len(classes) - n_ok} 类未齐全,"
                  f"但都在 P2 两条作业的覆盖范围内)")

    # ---- 冻结 / 对拍 ----
    if a.check:
        if not MANIFEST.exists():
            print(f"\n★ {MANIFEST} 不存在,先不带 --check 跑一次")
            return 2
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        drift = []
        for cls in classes:
            o, n = old["classes"].get(cls), man["classes"][cls]
            if o is None:
                drift.append(f"{cls}: 清单里没有,现在多了")
                continue
            for k, lbl in (("train_good", "train/good"), ("test_good", "test/good")):
                if o[k]["list_sha1"] != n[k]["list_sha1"]:
                    drift.append(f"{cls}/{lbl}: 列表哈希变了 "
                                 f"{o[k]['list_sha1']}→{n[k]['list_sha1']}")
            for d in set(o["defects"]) | set(n["defects"]):
                if d not in o["defects"]:
                    drift.append(f"{cls}/test/{d}: 新增缺陷类型")
                elif d not in n["defects"]:
                    drift.append(f"{cls}/test/{d}: 缺陷类型消失")
                elif o["defects"][d]["list_sha1"] != n["defects"][d]["list_sha1"]:
                    drift.append(f"{cls}/test/{d}: 列表哈希变了")
            if o.get("gt_allzero") != n.get("gt_allzero"):
                drift.append(f"{cls}: GT 全零清单变了")
        print("\n" + "=" * 92)
        if drift:
            print(f"★ 数据集相对冻结清单有 {len(drift)} 处漂移:")
            for d in drift:
                print(f"    {d}")
            return 1
        print("✓ 与冻结清单一致,无漂移")
        return 0

    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n冻结清单 → {MANIFEST}")

    print("\n" + "=" * 92)
    if warns:
        print(f"★ {len(warns)} 条警告:")
        for w in warns:
            print(f"    {w}")
    else:
        print("✓ 无警告:两套枚举一致、良品池内容哈希 15/15 互斥、GT 齐全、缓存无错配")
    print("\n★ 未覆盖:4i(用户 2026-09-12 定「暂时不跑 4i 相关的」;解冻时本节需补齐)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
