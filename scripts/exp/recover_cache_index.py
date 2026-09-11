"""反查良品缓存的**真实图号映射** —— 用 susp 字段,零神经网络。

## 起因

严格对拍(`parity_replay.py`)发现:缓存 `screw/000.npz` 与 `iter_test_images`
的第 0 张 test/good **逐像素一致(1.6e-7)**,但 `001.npz` 差了 4.4e-2。
同一段代码、同一个公式,索引 0 对得上而 1 对不上 ⇒ **不是公式错,是缓存顺序错了**。

`cache_good.py:93-100` 的写法:

    for ip in paths:
        f = d / f"{n:03d}.npz"
        if not f.exists():      # ← 已存在就跳过(幂等续跑)
            ...算并写...
        n += 1                  # ← 但 n 照增

只要**曾经用不同的 paths 顺序跑过一次**(日志里 cache_good{,2,3,15}.log 说明跑过多次),
残留文件就会被后续 run 当作"已完成的第 n 个"认领 —— 序号与图从此对不上。

## 为什么能反查

`susp` 是**从原图灰度直接算的**(`normalize_illumination` + `suspicion_map`,
纯 CV,不过网络),`cache_good.py:108-109`:

    cv = CvCfg(illum_norm="clahe")
    sus = patch_suspicion(suspicion_map(normalize_illumination(g, cv), cv), GRID)

对每张 test/good 原图重算这个 (15,15),与缓存里的 susp 做最近邻匹配即可。

## 这个错影响什么、不影响什么

  **不影响**本仓库所有 few-shot 实验:`shot_scaling` / `novelty_gate*` / `pixel_replay`
  只把良品当作**集合**(全用于标定、全用于 held-out),良品之间可互换,
  重排一个全是正常样本的集合不改变任何统计量。

  **影响**任何"缓存第 i 个 ↔ 图 i"的假设:可视化对照、逐图 spot check、
  以及严格对拍 —— 就是本次撞上的场景。

用法:
    python scripts/exp/recover_cache_index.py                 # 全部 15 类
    python scripts/exp/recover_cache_index.py screw,hazelnut
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402

import mvtec                                             # noqa: E402
from runtime.line.config import CvCfg                    # noqa: E402
from runtime.line.preprocess_cv import (                 # noqa: E402
    normalize_illumination, patch_suspicion, suspicion_map)
from runtime.pipeline import GRID                        # noqa: E402

CV = CvCfg(illum_norm="clahe")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="all")
    ap.add_argument("--data", default="data/mvtec_anomaly_detection")
    ap.add_argument("--good", default="/tmp/feat_cache_good")
    ap.add_argument("--out", default="data/deploy/good_cache_index.json")
    return ap.parse_args()


def susp_of(ip):
    g = np.asarray(Image.open(ip).convert("L"))
    return patch_suspicion(suspicion_map(normalize_illumination(g, CV), CV),
                           GRID).ravel()


def main() -> int:
    a = parse_args()
    classes = mvtec.MVTEC_CLASSES if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]

    print("反查良品缓存真实图号 | 匹配量 = susp(纯 CV,不过网络)")
    print(f"  数据 {a.data}   缓存 {a.good}")
    print()
    print(f"{'类':13s} {'缓存':>5s} {'test/good':>9s} {'双射':>5s}"
          f" {'匹配误差max':>12s} {'错位数':>7s}")
    print("-" * 62)

    out = {}
    for cls in classes:
        d = Path(a.good) / cls
        if not d.is_dir():
            print(f"{cls:13s} 无缓存,跳过")
            continue
        cf = sorted(d.glob("[0-9]*.npz"))
        picks = [ip for _n, rel, ip, _m in
                 mvtec.iter_test_images(a.data, cls) if rel == "good"]
        if not cf or not picks:
            print(f"{cls:13s} 文件 {len(cf)} / 图 {len(picks)},跳过")
            continue

        cache_s = np.stack([np.load(f)["susp"].ravel() for f in cf])
        img_s = np.stack([susp_of(ip) for ip in picks])
        # 每张缓存文件找最近的图
        D = np.linalg.norm(cache_s[:, None, :] - img_s[None, :, :], axis=2)
        best = D.argmin(axis=1)
        err = D[np.arange(len(cf)), best]

        bijection = len(set(best.tolist())) == len(cf)
        n_wrong = int((best != np.arange(len(cf))).sum())
        print(f"{cls:13s} {len(cf):5d} {len(picks):9d} "
              f"{'是' if bijection else '否':>5s} {err.max():12.2e} {n_wrong:7d}")
        out[cls] = {"n": len(cf), "bijection": bijection,
                    "map": [int(x) for x in best],
                    "err_max": float(err.max())}

    print("-" * 62)
    bad = [c for c, v in out.items() if not v["bijection"]]
    wrong = [(c, sum(1 for i, m in enumerate(v["map"]) if i != m))
             for c, v in out.items()]
    print(f"非双射的类: {bad if bad else '无'}")
    print(f"错位的类: " + (", ".join(f"{c}({n})" for c, n in wrong if n)
                          if any(n for _c, n in wrong) else "无"))
    print()
    print("★ 错位**不影响** few-shot 统计(良品是集合,可互换);")
    print("  但任何'缓存第 i 个 ↔ 图 i'的假设都要用本映射纠正。")
    if out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\n映射已写 {a.out}(类 → 缓存下标 → 真实图号)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
