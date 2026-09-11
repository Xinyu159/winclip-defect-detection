"""任务 B:int8 端到端量化漂移表(逐类 Δ)。

任务书 T01 §3 规定的列:
    类 | fp32 img/pix | int8 img/pix | Δimg | Δpix | 是否超 ±0.3pt

两个口径都要出:zero-shot(shots=0)与 few-shot(shots=4)。
**同一硬件**对比(CUDA vs CUDA)—— 混硬件的话差值就归因不清了。

用法:
    python scripts/make_t01_B_table.py \
        --fp32 logs/ov_*_fp32_cuda.json --int8 logs/ov_*_int8_cuda.json \
        --out 发件箱/T01_B_量化漂移表.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 报告阈值:漂移超过这个绝对值要在表里标红(任务书规定 0.3pt)
TOL = 0.3


def load(p: str) -> dict:
    d = json.loads(Path(p).read_text())
    return {sh: v["per_class"] for sh, v in d["results"].items()}


def cell(delta: float) -> str:
    mark = " ⚠" if abs(delta) > TOL else ""
    return f"{delta:+.1f}{mark}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    F, I = load(args.fp32), load(args.int8)
    fp32_meta = json.loads(Path(args.fp32).read_text())
    int8_meta = json.loads(Path(args.int8).read_text())

    L = ["# T01-B int8 量化端到端漂移", ""]
    L += ["## 0. 测量条件", "",
          f"- fp32:`{fp32_meta['deploy']}` | {fp32_meta['model_sizes_mb']}",
          f"- int8:`{int8_meta['deploy']}` | {int8_meta['model_sizes_mb']}",
          f"- 引擎/设备(fp32):`{fp32_meta['engine']}` / "
          f"`{fp32_meta['device']}`",
          f"- 引擎/设备(int8):`{int8_meta['engine']}` / "
          f"`{int8_meta['device']}`",
          f"- 数据:{len(next(iter(F.values())))} 类,全部缺陷图 + 全部良品",
          f"- 生成时间:fp32 `{fp32_meta['time']}` / int8 `{int8_meta['time']}`",
          ""]
    if fp32_meta["device"] != int8_meta["device"]:
        L += ["> ⚠ **两档设备不一致**,差值无法归因于量化。", ""]
    L += [f"漂移 = int8 − fp32。绝对值 > {TOL}pt 标 ⚠。"
          "负值代表 int8 更差(正常);正值代表 int8 反而更好(通常是噪声)。", ""]

    for shot in sorted(set(F) & set(I)):
        f, i = F[shot], I[shot]
        lab = "zero-shot" if shot == "0" else f"{shot}-shot"
        L += [f"## {'一' if shot == '0' else '二'}、{lab}(shots={shot})", "",
              "| 类 | fp32 img | int8 img | Δimg | fp32 pix | int8 pix | Δpix | 超 ±0.3pt |",
              "|---|---|---|---|---|---|---|---|"]
        di, dp, over = [], [], []
        for cls in sorted(f):
            if cls not in i:
                L.append(f"| {cls} | — | — | — | — | — | — | (int8 缺该类) |")
                continue
            a, b = f[cls], i[cls]
            d1 = b["img_auroc"] - a["img_auroc"]
            d2 = b["pix_auroc"] - a["pix_auroc"]
            di.append(d1)
            dp.append(d2)
            bad = abs(d1) > TOL or abs(d2) > TOL
            if bad:
                over.append((cls, d1, d2))
            L.append(f"| {cls} | {a['img_auroc']:.1f} | {b['img_auroc']:.1f} | "
                     f"{cell(d1)} | {a['pix_auroc']:.1f} | {b['pix_auroc']:.1f} | "
                     f"{cell(d2)} | {'**是**' if bad else '否'} |")
        n = len(di)
        L.append(f"| **均值** | — | — | **{sum(di)/n:+.2f}** | — | — | "
                 f"**{sum(dp)/n:+.2f}** | — |")
        L.append("")
        if over:
            L += [f"**超阈值类({len(over)} 个)**:", ""]
            for cls, d1, d2 in over:
                L.append(f"- `{cls}`:Δimg {d1:+.1f}pt,Δpix {d2:+.1f}pt")
            L.append("")
        else:
            L += ["**无类超过阈值。**", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
