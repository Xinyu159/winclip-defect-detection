"""把分片跑的 bank_arms.py 结果合并成 15 类主表。

分片是因为单进程跑满 15 类要 ~60 分钟,而各类成本差异很大(≈ (缺陷数+eval张数)×库张数,
最贵的 cable 是最便宜的 toothbrush 的 16 倍)。按成本配平切 3 片,墙钟时间降到 1/3。

**合并不是简单平均**:各片只知道自己那几类的均值,`15类均值` 与 `胜/平/负` 计数
必须跨片重算。本脚本做的就是这个。

用法:
    python scripts/exp/merge_arms_shards.py /tmp/bank_arms_p1_1.json /tmp/bank_arms_p1_2.json ...
"""
import json
import sys
from pathlib import Path

import numpy as np

ALL15 = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
METRICS = ("px_prod", "px_bank", "img")


def main(paths) -> int:
    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print(f"合并 {len(shards)} 片:" + "".join(
        f"\n   {p}  {len(s['classes'])} 类 reps={s['reps']} pool={s['pool']}"
        for p, s in zip(paths, shards)))

    classes, n_img = [], {}
    for s in shards:
        for c in s["classes"]:
            if c in n_img:
                raise SystemExit(f"★ 类 {c} 出现在多片里,合并会重复计数")
            classes.append(c)
            n_img[c] = s["n"][c]

    # ★ 臂表随每个类的库大小变(G_k 只在 k<=n_bank 时建),所以各片取交集 ——
    # 不能因为一片没 G_8 就拒绝合并整张表。被剔的臂明确报出来,不静默丢。
    order = list(shards[0]["arms"])
    common = set.intersection(*[set(s["arms"]) for s in shards])
    arms = [a for a in order if a in common]
    for a in sorted(set.union(*[set(s["arms"]) for s in shards]) - common):
        who = [f"{p}({len(s['classes'])}类)" for p, s in zip(paths, shards)
               if a not in s["arms"]]
        print(f"   ★ 臂 {a} 非每片都有,已从主表剔除;缺它的片: {', '.join(who)}")
    if not arms:
        raise SystemExit("★ 各片没有任何公共臂,合并没有意义")

    # 各片的 arms[a][m] 是按该片 classes 顺序对齐的,直接首尾相接
    val = {m: {a: [] for a in arms} for m in METRICS}
    for s in shards:
        for a in arms:
            for m in METRICS:
                val[m][a].extend(s["arms"][a][m])

    missing = [c for c in ALL15 if c not in classes]
    order = [c for c in ALL15 if c in classes] + [c for c in classes if c not in ALL15]
    idx = [classes.index(c) for c in order]

    def get(a, m):
        return np.array(val[m][a])[idx]

    print(f"\n类数 {len(classes)}" + (f"  ★ 缺 {missing}" if missing else "  (15/15 齐)"))
    print(f"\nreps={shards[0]['reps']} seed={shards[0]['seed']} pool={shards[0]['pool']}")
    print(f"{'类':12s}{'库':>5s}{'eval':>6s}{'缺陷':>6s}")

    print("\n" + "=" * 100)
    print("主结果 pixel AUROC — map_prod(生产图)| 逐类 = reps 均值")
    print("=" * 100)
    print(f"{'臂':9s}" + "".join(f"{c[:9]:>9s}" for c in order) +
          f"{'均值':>9s}{'vs G_4':>9s}{'胜/平/负':>11s}")
    ref = get("G_4", "px_prod")
    for a in arms:
        v = get(a, "px_prod")
        mu = float(v.mean())
        if a == "G_4":
            dn = wl = ""
        else:
            d = v - ref
            wl = (f"{int((d > 1).sum()):2d}/{int((abs(d) <= 1).sum()):2d}/"
                  f"{int((d < -1).sum()):2d}")
            dn = f"{mu - ref.mean():+9.1f}"
        print(f"{a:9s}" + "".join(f"{x:9.1f}" for x in v) +
              f"{mu:9.1f}{dn:>9s}{wl:>11s}")
    print("\n（胜/平/负:相对 G_4 的逐类配对差,门限 ±1pt;§五 判据要求 ≥8/15 为正 且 均值 ≥ +2pt）")

    print("\n" + "=" * 100)
    print("副表:map_bank(纯库图像素)| img(image AUROC)")
    print("=" * 100)
    print(f"{'臂':9s}{'map_bank 均值':>16s}{'img 均值':>12s}")
    for a in arms:
        print(f"{a:9s}{get(a, 'px_bank').mean():16.1f}{get(a, 'img').mean():12.1f}")

    out = {"pool": shards[0]["pool"], "reps": shards[0]["reps"],
           "seed": shards[0]["seed"], "classes": order,
           "n": {c: n_img[c] for c in order},
           "arms": {a: {m: get(a, m).tolist() for m in METRICS} for a in arms}}
    p = Path("/tmp/bank_arms_merged.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n合并结果 → {p}")

    # ---- §五 判据的直接判决 ----
    print("\n" + "=" * 100)
    print("§五 判据:B 系库臂相对现行 G_4 的 pixel AUROC 配对提升")
    print("=" * 100)
    for a in arms:
        if a in ("G_4", "Z0") or a.startswith("G_") or a.startswith("A_r"):
            continue
        d = get(a, "px_prod") - ref
        ok = d.mean() >= 2.0 and int((d > 0).sum()) >= 8
        print(f"  {a:10s} 均值 {d.mean():+5.2f}pt  为正 {int((d>0).sum())}/{len(d)}  "
              f"⇒ {'成立' if ok else '不成立'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
