"""出 Surface Defects-4i 的 WinCLIP 数字表(给 C++ 传统方法侧对照用)。

输入三份 evaluate.py 日志:
  --main    12 类 × shots{0,1,4}  的主表
  --mvtec200 MVTec tile/leather 降到 200px 的消融(隔离"分辨率"因素)
  --gt128    GT 二值化阈值 >0→>128 的消融(隔离"真值口径"因素)

用法:
    python scripts/make_4i_table.py --main logs/4i_exp.json \
        --mvtec200 logs/4i_abl_mvtec200.json --gt128 logs/4i_abl_gt128.json \
        --out results/4i_winclip_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 按材料分组排列(同一"物体名"的类相邻),便于看出哪些类共用 CPE 文本
ORDER = ["Al_Rm", "Al_Con", "MT_Uneven", "MT_Break", "MT_Fray",
         "Steel_Ld", "Steel_Am", "Steel_Pa", "Steel_Sc",
         "Rail", "Leather", "Tile"]

# 已发表的 MVTec 15 类表里 tile/leather 的 zero-shot 数字(T01_B_table.md,同硬件同权重)
MVTEC_REF = {"tile": (99.9, 75.2), "leather": (100.0, 96.3)}


def cell(r):
    return f"{r['img_auroc']:.1f} / {r['pix_auroc']:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--mvtec200", default="")
    ap.add_argument("--gt128", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.main).read_text())
    res = d["results"]
    shots = sorted(res.keys(), key=int)

    L = ["# Surface Defects-4i × WinCLIP 数字表", "",
         "> 供 `C++传统版4i升级手册.md` 的传统方法侧对照。**同一数据集、同一划分、同一指标口径。**", "",
         "## 0. 协议", "",
         f"- **权重**:`{Path(d['weights']).name}`(ViT-B-16-plus-240,LAION-400M E31)",
         f"- **设备**:`{d['device']}` | **seed**:`{d['seed']}` | **脚本**:`{d['script']}`",
         f"- **CPE 物体名**:`{d['prompt_map']}`(由 `make_4i_mvtec.py` 生成,非手写)",
         "- **指标**:`sklearn.roc_auc_score`;image 级 = CLIP 整图,"
         "pixel 级 = 异常图上采样到 GT 后逐像素展平",
         "- **划分**(`scripts/make_4i_mvtec.py`,seed=42):",
         "  - `train/good` = Nd 画廊(few-shot 用),`test/good` = 留出的 Nd,**两者互斥**",
         "  - `test/good` 抽 `min(50, N//2)`;小池类(MT_Fray 31、MT_Uneven 55、MT_Break 85)"
         "因池子不够只能少抽",
         "  - `test/<缺陷类型>` = 该类全部 Images",
         "- **GT 口径**:转换期按 `>0` 落成 {0,255}。`evaluate.py` 下游的 `>128` 因此**恒等** —— "
         "于是与 C++ 手册 §2「GT>0」是**同一个真值定义**(见 §3.2 的敏感性实测)", ""]

    # ---- 主表 ----
    L += ["## 1. 主表(image / pixel,evaluate.py 产出)", "",
          "| 类 | 缺陷 n | test/good n | " +
          " | ".join(f"**{s}-shot**" for s in shots) + " |",
          "|---|---|---|" + "---|" * len(shots)]
    for c in ORDER:
        if c not in res[shots[0]]["per_class"]:
            continue
        r0 = res[shots[0]]["per_class"][c]
        L.append(f"| `{c}` | {r0['n_defect']} | {r0['n_test'] - r0['n_defect']} | " +
                 " | ".join(cell(res[s]["per_class"][c]) for s in shots) + " |")
    # 直接用 evaluate.py 存进 json 的均值,而不是拿逐类**已四舍五入**的值再平均
    # —— 后者会因舍入方向不同与日志差 0.1(1-shot 实测 94.7 vs 日志 94.6)
    L.append("| **12 类均值** | — | — | " +
             " | ".join(f"**{res[s]['mean_img_auroc']:.1f} / "
                        f"{res[s]['mean_pix_auroc']:.1f}**"
                        for s in shots) + " |")
    L += ["", "每格为 `image AUROC / pixel AUROC`(百分制)。", ""]

    # ---- 反常结果,不平滑 ----
    L += ["## 2. 反常/未达预期的结果(如实报出,未做平滑)", ""]
    pc0, pc4 = res[shots[0]]["per_class"], res[shots[-1]]["per_class"]
    below = [c for c in ORDER if c in pc0 and pc0[c]["pix_auroc"] < 50]
    if below:
        L += ["### 2.1 pixel AUROC **低于 50%(不如抛硬币)**", "",
              "| 类 | " + " | ".join(f"{s}-shot pix" for s in shots) + " |",
              "|---|" + "---|" * len(shots)]
        for c in below:
            L.append(f"| `{c}` | " + " | ".join(
                f"**{res[s]['per_class'][c]['pix_auroc']:.1f}**" for s in shots) + " |")
        L += ["", "→ 分数与真值**反相关**:模型在正常像素上给的分比缺陷像素还高。"
              "这不是「定位不准」,是方向错了。**MT_Uneven 三档全在 50 以下**(38.0 / 46.1 / 48.4),"
              "few-shot 只把它从 38 抬到 48,仍没过线。", ""]

    nonmono = [c for c in ORDER
               if c in pc0 and not (pc0[c]["pix_auroc"] <= res["1"]["per_class"][c]["pix_auroc"]
                                    <= pc4[c]["pix_auroc"])]
    if nonmono:
        L += ["### 2.2 few-shot 增益**非单调**", "",
              "| 类 | " + " | ".join(f"{s}-shot pix" for s in shots) + " | 形态 |",
              "|---|" + "---|" * len(shots) + "---|"]
        for c in nonmono:
            v = [res[s]["per_class"][c]["pix_auroc"] for s in shots]
            shape = "↑↓ 掉头" if v[1] > v[0] and v[2] < v[1] else "↓↑ 先降后升"
            L.append(f"| `{c}` | " + " | ".join(f"{x:.1f}" for x in v) + f" | {shape} |")
        L += ["", "→ few-shot 不是单调变好。**Al_Rm 甚至 0→1shot 涨 10.5 点后又回落**,"
              "说明 gallery 采样(1 张!)带来的方差不可忽略。"
              "样本少的类尤其不要把单档数字当稳定结论。", ""]

    # ---- 消融 ----
    L += ["## 3. 两处归因消融(都把「疑似原因」测掉了)", ""]
    if args.mvtec200:
        m = json.loads(Path(args.mvtec200).read_text())["results"]["0"]["per_class"]
        L += ["### 3.1 怀疑「200px 分辨率」→ **证伪**", "",
              "4i 的 tile/leather 与 MVTec **共用同一批正常图**(见 §4),"
              "但 pixel AUROC 差很多,于是怀疑是 4i 只有 200px 所致。"
              "把 MVTec 原图整体降到 200px 重测:", "",
              "| 类 | MVTec@1024 | MVTec@200 | **4i@200** | 1024→200 掉了 | 200 之外剩下的差 |",
              "|---|---|---|---|---|---|"]
        for k, c in (("tile", "Tile"), ("leather", "Leather")):
            a = MVTEC_REF[k][1]
            b = m[k]["pix_auroc"]
            c4 = pc0[c]["pix_auroc"]
            L.append(f"| {k} | {a:.1f} | {b:.1f} | **{c4:.1f}** | "
                     f"{a-b:+.1f} | {b-c4:+.1f} |")
        L += ["", "→ **分辨率不是主因**:leather 降到 200px 几乎没掉(96.3→96.6),"
              "4i 却低了 16 点。tile 分辨率只解释 7.3 点,剩下 12.3 点解释不了。"
              "**差距只能归到 4i 的缺陷图/掩码本身** —— 且 §4 已证明 4i 的缺陷图确实不是 MVTec 的那批。", ""]
    if args.gt128:
        g = json.loads(Path(args.gt128).read_text())["results"]["0"]["per_class"]
        L += ["### 3.2 怀疑「GT 二值化阈值」→ **影响 <1 点**", "",
              "4i 有 5 个类的 GT 是 0~255 连续灰阶(像高斯糊过的边界),"
              "另有 7 个类是硬 0/255。担心 `>0` 把边界光晕也算成缺陷:", "",
              "| 类 | GT `>0` | GT `>128` | Δ |", "|---|---|---|---|"]
        for c in ("Tile", "Leather"):
            a, b = pc0[c]["pix_auroc"], g[c]["pix_auroc"]
            L.append(f"| {c} | {a:.2f} | {b:.2f} | **{b-a:+.2f}** |")
        L += ["", "→ 两种口径差 **<1 点**,不构成结论差异。"
              "**C++ 侧用 `>0` 或 `>128` 都能对上**,不必为此返工。"
              "(光晕占比本身在 `logs/4i_manifest.json` 里有逐类记录,"
              "Leather 15.3% / MT_Break 15.6% / Tile 8.6% / 其余硬标签类 0%)", ""]

    # ---- §4 同源 ----
    L += ["## 4. ★ Leather/Tile 与已发表的 MVTec 表**同源**,不可当独立证据", "",
          "论文参考文献 [39] 就是 MVTec AD 论文,4i 的 Leather/Tile 取自 MVTec。实测:", "",
          "| 4i 类 | 4i Nd 张数 | MVTec train/good | MVTec test/good | 合计 |",
          "|---|---|---|---|---|",
          "| Leather | 277 | 245 | 32 | **277** |",
          "| Tile | 263 | 230 | 33 | **263** |",
          "",
          "两个类都**逐个对上**,且最近邻匹配的**去重命中数恰好等于 train/good 的张数**"
          "(245 / 230),全部 Nd 的相关系数 ≥0.99。→ **4i 的 Nd 就是 MVTec 的 train/good ∪ test/good。**",
          "",
          "但 **4i 的缺陷图不是 MVTec 的那批**:对 MVTec 全部缺陷图做最近邻,"
          "最高相关只有 0.23(tile)/ 0.73(leather),**无一超过 0.99**。", "",
          "**后果**:我们已发表的 MVTec 15 类表里含 `leather`/`tile` 两行,"
          "与 4i 的这两行**共用同一批正常图**。→ 两张表在这两行上**不是相互独立的证据**,"
          "README 里必须写明,不能摆成「两个数据集都验证了」的样子。", ""]

    # ---- §5 ----
    L += ["## 5. 已知限制(不许编数字)", "",
          "1. **良品样本极少限制过杀率精度**。`test/good` 只有 15~50 张"
          "(MT_Fray 15、MT_Uneven 27),过杀率的最小非零粒度 = 1/N。"
          "**本表不报任何 1% 量级的过杀率。**",
          "2. **剔除了 1 张源数据自带的全零 GT**(`MT_Break_8.png`,GT 像素全 0、无标注)。"
          "留着会让它的高分像素被当成假阳、**压低** MT_Break 的 pixel AUROC。"
          "故 MT_Break 缺陷数为 **56**(手册记的 57 含这一张)。",
          "3. **剔除是在转换期做的**,`logs/4i_manifest.json` 的 `dropped_empty_gt` 字段有记录。"
          "**C++ 侧请同样剔除**,否则两边差一张图。",
          "4. **image AUROC 普遍接近满分**(12 类均值 93.7→94.8),分辨力有限;"
          "这张表真正的信息量在 **pixel AUROC**。",
          "5. 本表**只报现状,不提供调优后的模型**;未做任何用 GT 选参的动作。", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
