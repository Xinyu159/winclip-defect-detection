"""int8 掉点归因:排序翻转噪声 vs 真量化损伤(**本地零推理,不碰远程**)。

## 问题

T01-B 实测(int8 − fp32,同机同 EP,CUDA)15 类均值 image **−1.15pt** / pixel −0.65pt,
最差类 toothbrush −6.1、transistor −4.5。

但**反证同样在表里**:zipper **+1.9**、wood +0.4、tile +0.1、leather 0.0 ——
int8 有的类**反而更好**。若 int8 是系统性损伤,正号不该出现。
⇒ −1.15pt 里混着多少「排序翻转噪声」、多少「真损伤」? 没拆开之前,
"改量化层"可能是在修一个不存在的问题。

## 已知的噪声底(两端都有,中间缺)

| 特征余弦 | 实测逐类最大 |Δ| | 出处 |
|---|---|---|---|
| ~1.0(fp32 纯浮点差) | **0.10pt** | 本地 OV-CPU vs 远程 ONNX-CUDA,2026-09-14 |
| **0.995–0.997(int8)** | **?** | ← 本脚本补的就是这一格 |

int8 模型在远程、本地没有,但**特征缓存在**(`artifacts/`,本地 OV 塔建的)。
给缓存特征加噪声把余弦调到 int8 实测量级,走同一条打分链重复 R 次,
得到每类 AUROC 的**噪声分布**;把 int8 的实测 Δ 落到这个分布上。

## 判据(跑之前写死)

- |Δ_int8| ≤ 2σ(该类噪声分布) ⇒ 该类的"掉点"**与噪声不可分**,不是量化损伤
- 显著超出 2σ            ⇒ **真损伤**,值得针对性改量化层

## ★ 诚实边界(三条,别越界)

1. 这是**灵敏度标定**,不是 int8 的复现。它量的是「打分链对特征扰动的敏感度」,
   扰动**幅度**取自 int8 实测余弦(int8 侧的事实),扰动**分布形状**(各向同性高斯)
   是假设(int8 的真实量化误差不是高斯)。标记为「我们的设计选择」,不作 int8 证据。
2. **跨平台的一次标定转移**:噪声底用**本地 OV** 特征标定,而 Δ_int8 是**远程 ONNX**
   上量的 —— 按平台分离铁律,两者是不同平台。允许这么做的理由(是假设,不是结论):
   扰动敏感度由**打分数学 + 该类的近分对数**决定,与哪个引擎产出特征无关;且两平台
   特征此前已证到 0.1pt 精度下 AUROC 不可分。**若判出"真损伤",结论仍须在远程复现验证。**
3. **靶用本地 OV log**(`logs/ov_20260914_004801_fp32_localov.json`),不用远程 ——
   缓存是本地 OV 建的,同平台才能构成对照。远程那份只作旁证打印出来(见下)。

用法:
    .venv/bin/python scripts/exp/int8_noise_floor.py                 # 15 类,默认 4 档余弦
    .venv/bin/python scripts/exp/int8_noise_floor.py --reps 30 --cos 0.996
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts" / "exp") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "exp"))

from pixel_replay import ALL15, load_npz            # noqa: E402
from runtime.pipeline import OVPipeline             # noqa: E402

FEATS = ("full", "w3", "w5")                        # 三个塔的输出,全都要扰动


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="all")
    ap.add_argument("--good", default="artifacts/feat_cache_good")
    ap.add_argument("--bad", default="artifacts/feat_cache")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--cos", default="0.999,0.997,0.996,0.995",
                    help="目标余弦,逗号分隔(1.0 = 无扰动对照)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/exp/int8_noise_floor.json")
    ap.add_argument("--ref", default="logs/ov_20260914_004801_fp32_localov.json",
                    help="对照靶:必须与特征缓存同平台(本地 OV)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="对照容差。归档 log 只留 1 位小数 ⇒ 落档本身就带 0.05 量化,"
                         "0.10 是「两次完全相同的计算仍可能差到」的最小上界")
    return ap.parse_args()


def stack_feats(imgs: list) -> dict:
    """[npz…] → {feat: (N, L, 640)}。

    ★ 必须 `np.stack` 而不是 `np.concatenate` —— 后者会把 (226,640) 沿 0 轴
    摞成 (N·226, 640),把「图」维和「token」维混掉,而且**不会报错**:
    打分退化成「逐 token 打分」,得到的仍是合法 AUROC,只是含义完全不同。
    (2026-09-14 首版正是这么错的,bottle/screw 两类的对照值就对不上。)
    """
    return {f: np.stack([z[f] for z in imgs], axis=0) for f in FEATS}


def auc_once(D: dict, cos: float, rng: np.random.Generator):
    """扰动一次 → (image AUROC, 实测余弦)。cos>=1.0 = 不扰动对照。"""
    scores, coss = [], []
    for key in ("good", "bad"):
        raw = D[key]
        cur = {}
        for f in FEATS:
            arr = raw[f]                                  # (N,L,640)
            if cos >= 1.0:
                cur[f] = arr
                continue
            d = arr.shape[-1]
            sigma = math.sqrt(1.0 / (cos * cos) - 1.0) / math.sqrt(d)
            v = arr + (sigma * rng.standard_normal(arr.shape)).astype(arr.dtype)
            cur[f] = v / np.linalg.norm(v, axis=-1, keepdims=True)
        # CLS 行(full[:, :1])——与 pipeline.py 的 cls_prob 完全同一条
        scores.append(np.array([float(OVPipeline._prob(cur["full"][i, :1],
                                                       D["pos"], D["neg"],
                                                       D["temp"])[0])
                                for i in range(cur["full"].shape[0])]))
        coss.append(float(np.mean(np.sum(raw["full"] * cur["full"], axis=-1))))
    auc = roc_auc_score(np.r_[np.zeros(len(scores[0])), np.ones(len(scores[1]))],
                        np.r_[scores[0], scores[1]]) * 100
    return float(auc), float(np.mean(coss))


def main() -> int:
    a = parse_args()
    classes = ALL15 if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]
    cos_list = [float(x) for x in a.cos.split(",")]

    print("加载缓存…(读 npz 是最贵的一步,只做一次)", flush=True)
    data = {}
    for cls in classes:
        tp = Path(a.text) / f"{cls}.npz"
        good = load_npz(Path(a.good) / cls)
        bad = load_npz(Path(a.bad) / cls, only_gt=True)
        if not tp.exists() or not good or not bad:
            print(f"  {cls}: 文本原型或缓存缺失,跳过")
            continue
        d = np.load(tp)
        data[cls] = {"pos": d["normal"], "neg": d["abnormal"], "temp": float(d["temp"]),
                     "good": stack_feats(good), "bad": stack_feats(bad)}

    fp32 = json.load(open("logs/ov_20260910_153244_fp32_cuda.json"))["results"]["0"]["per_class"]
    int8 = json.load(open("logs/ov_20260910_154500_int8_cuda.json"))["results"]["0"]["per_class"]

    # ── 对照:无扰动必须逐类复现归档 fp32(否则后面全不成立)──────
    ref_log = json.load(open(a.ref))["results"]["0"]["per_class"]
    print(f"\n无扰动对照(靶={a.ref},本地 OV,与缓存同平台)")
    print(f"{'类':12s}{'重放':>8s}{'本地OV':>8s}{'Δ':>7s}{'远程CUDA':>10s}{'跨平台Δ':>9s}")
    base, bad = {}, []
    for cls, D in data.items():
        auc, _ = auc_once(D, 1.0, np.random.default_rng(0))
        base[cls] = round(auc, 4)
        r_loc = ref_log[cls]["img_auroc"]
        r_rem = fp32[cls]["img_auroc"]
        flag = "  ✗" if abs(auc - r_loc) > a.tol else ""
        bad += [cls] if abs(auc - r_loc) > a.tol else []
        print(f"{cls:12s}{auc:8.2f}{r_loc:8.2f}{auc-r_loc:+7.2f}"
              f"{r_rem:10.2f}{r_loc-r_rem:+9.2f}{flag}")
    if bad:
        print(f"\n✗ 对照不通过:{bad}(容差 {a.tol})—— 先定位,不往下跑")
        return 1
    print(f"  ✓ {len(data)} 类逐类吻合(容差 {a.tol};落档 1 位小数自带 0.05 量化)")

    report = {"script": "int8_noise_floor.py", "reps": a.reps, "cos_list": cos_list,
              "seed": a.seed, "baseline_image": base,
              "int8_delta": {c: round(int8[c]["img_auroc"] - fp32[c]["img_auroc"], 2)
                             for c in data},
              "levels": {}}

    for cos in cos_list:
        per_cls = {}
        for cls, D in data.items():
            rng = np.random.default_rng(a.seed)
            res = [auc_once(D, cos, rng) for _ in range(a.reps)]
            aucs = [r[0] for r in res]
            per_cls[cls] = {"mean": round(float(np.mean(aucs)), 2),
                            "std": round(float(np.std(aucs, ddof=1)), 2),
                            "achieved_cos": round(float(np.mean([r[1] for r in res])), 5)}
        report["levels"][str(cos)] = per_cls
        print(f"\n[cos={cos}] 实测余弦 "
              f"{np.mean([v['achieved_cos'] for v in per_cls.values()]):.5f}"
              f"  → 各类 AUROC 的 σ:")
        print("  " + "  ".join(f"{c[:9]} {per_cls[c]['std']:.2f}" for c in data))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[log] {out}")

    # ── 判决 ────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("判决:|Δ_int8| ÷ 该类的噪声 σ。>2σ = 与噪声不可分之外 ⇒ 真损伤")
    print("=" * 84)
    print(f"{'类':12s}{'Δ_int8':>8s}" +
          "".join(f"{('Δ/σ@'+str(c)[2:]):>12s}" for c in cos_list) + f"{'最大倍数':>10s}")
    rows = []
    for cls in data:
        d = report["int8_delta"][cls]
        zs = [abs(d) / max(report["levels"][str(c)][cls]["std"], 1e-6) for c in cos_list]
        rows.append((max(zs), cls, d, zs))
        print(f"{cls:12s}{d:+8.2f}" + "".join(f"{z:11.2f}" for z in zs)
              + f"{max(zs):10.2f}")
    rows.sort(reverse=True)
    real = [r for r in rows if r[0] >= 2.0]
    print(f"\n★ 真损伤候选(≥2σ):" +
          ("  ".join(f"{c} {z:.1f}σ (Δ{d:+.1f})" for z, c, d, _ in real) or "(无)"))
    print(f"★ 与噪声不可分(<2σ):{len(rows)-len(real)}/{len(rows)} 类")
    print("\n注:σ 是「特征扰动到 int8 同量级余弦时,该类 AUROC 的抖动」;"
          "判决读的是 Δ 与 σ 的比,不是 Δ 的绝对大小。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
