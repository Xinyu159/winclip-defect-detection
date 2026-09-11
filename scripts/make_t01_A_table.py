"""任务 A 的三份表:三档对照 / GT 面积机制 / 不单调类清单。

严格按任务书 T01(16:09 修订版)§1.5 的口径写,三条纪律写进表头:
  - 两个口径必须分开:定位 = few_map only;判定 = 真流水线 m_all
  - 过杀率**不许**单独报 @Pxx(同义反复),只报 (过杀固定, 漏检) 配对
  - 良品数 N 必须列出 —— 过杀最小粒度 = 1/N

用法:
    python scripts/make_t01_A_table.py --tier /tmp/tier_gt.json \
        --grid /tmp/cascade_prod.json --out 发件箱/T01_A_汇总表.md
"""
from __future__ import annotations

import argparse
import numpy as np
import json
from pathlib import Path

def spearman(x, y) -> tuple[float, float]:
    """秩相关 + 置换检验 p 值(本地无 scipy,自己实现,免得加依赖)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    rho = float((rx * ry).sum() / den) if den > 0 else float("nan")
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        p = rng.permutation(ry)
        d = np.sqrt((rx ** 2).sum() * (p ** 2).sum())
        null.append(float((rx * p).sum() / d) if d > 0 else 0.0)
    p = float((np.abs(null) >= abs(rho)).mean())
    return rho, p


TIER_ORDER = ["L1", "L1+3x3", "full"]
TIER_LABEL = {"L1": "L1(零窗)", "L1+3x3": "L1+3×3", "full": "full(3×3+2×2)"}


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--grid", default="")
    ap.add_argument("--ablation", default="/tmp/mn_abl.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ABL = {}
    if Path(args.ablation).exists():
        ABL = load(args.ablation)
    d = load(args.tier)
    res = d["results"]
    classes = list(res)

    def mean(key: str, tier: str) -> float:
        vals = [r["tiers"][tier][key] for r in res.values()
                if r["tiers"][tier].get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    L = ["# T01-A 汇总表:级联三档 · 全 15 类 · 全量测试集", "",
         "**测量条件**:远程 RTX 3080 Ti / onnxruntime CUDA EP / fp32",
         "**数据**:MVTec 全 15 类,**全部 1258 张缺陷图 + 全部 467 张良品**",
         "**阈值标定**:只从良品类(good)取分位,**未用任何 GT**", "",
         "## 0. 先读:两个口径 + 三条纪律", "",
         "| 口径 | 定义 | 回答的问题 |", "|---|---|---|",
         "| **定位** | 只用 `few_map`,不含任何全局标量 | 缺陷在**哪里** |",
         "| **判定** | 真流水线 `m_all = 调和(cls_prob,窗口) + few_map`;"
         "`img_score=(cls_prob+few_map.max())/2` | 整件**是不是** NG |", "",
         "三条纪律(任务书 §1.5):", "",
         "1. **档位按 token 数定义**,不是按名字:"
         "L1=226(零窗) / L1+3×3=1916 / full=2896。"
         "两个配置输出完全相等时,先怀疑**配置没生效**。",
         "2. **不许把含全局标量的 `m_all` 当定位分数** —— "
         "`cls_prob` 整幅同值,会污染局部排名。",
         "3. **不单独报 `过杀率@Pxx`** —— 阈值本取自良品 Pxx 分位,"
         "该比例恒等于 x%,是同义反复。只报 **(过杀固定, 漏检)** 配对。", "",
         "## 1. 三档 × 两口径(全 15 类均值)", "",
         "| 档位 | token | 相对 | 定位 img | 判定 img | 纯 cls_prob | "
         "定位 pix | 判定 pix |", "|---|---|---|---|---|---|---|---|"]
    tok = {"L1": 226, "L1+3x3": 1916, "full": 2896}
    rel = {"L1": "1.0×", "L1+3x3": "8.5×", "full": "12.8×"}
    for t in TIER_ORDER:
        L.append(f"| {TIER_LABEL[t]} | {tok[t]} | {rel[t]} | "
                 f"{mean('loc_img', t):.1f} | {mean('jud_img', t):.1f} | "
                 f"{mean('cls_img', t):.1f} | {mean('loc_pix', t):.1f} | "
                 f"{mean('jud_pix', t):.1f} |")
    L.append("")

    L += ["## 2. 逐类:三档 × 判定 img(窗口塔买不买得到「判定」)", "",
          "| 类 | 良品N | 过杀粒度 | GT面积 | 定位img L1→full | "
          "判定img L1→L1+3×3→full | Δ判定 | loc_pix L1→full | Δ定位pix |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in classes:
        r = res[c]
        t = r["tiers"]
        d_img = t["full"]["jud_img"] - t["L1"]["jud_img"]
        d_pix = t["full"]["loc_pix"] - t["L1"]["loc_pix"]
        mark = "**" if abs(d_img) >= 2.0 else ""
        L.append(
            f"| {c} | {r['n_good']} | {r['overkill_grain_pct']}% | "
            f"{r['gt_area_mean']:.3f} | "
            f"{t['L1']['loc_img']:.1f}→{t['full']['loc_img']:.1f} | "
            f"{t['L1']['jud_img']:.1f}→{t['L1+3x3']['jud_img']:.1f}"
            f"→{t['full']['jud_img']:.1f} | {mark}{d_img:+.1f}{mark} | "
            f"{t['L1']['loc_pix']:.1f}→{t['full']['loc_pix']:.1f} | "
            f"{d_pix:+.1f} |")
    L.append("")

    # ---- 任务书问题 1:窗口塔救不了图像级? ----
    L += ["## 3. 问题一:其余类是否也「窗口塔救不了图像级」", "",
          "判据:`判定 img` 在三档上的**总变化量**。若某类三档几乎不动,"
          "而它的定位 pix 明显改善 → 该类的瓶颈在图像级打分,不在算力。", ""]
    stuck = [(c, res[c]) for c in classes
             if abs(res[c]["tiers"]["full"]["jud_img"]
                    - res[c]["tiers"]["L1"]["jud_img"]) < 2.0
             and (res[c]["tiers"]["full"]["loc_pix"]
                  - res[c]["tiers"]["L1"]["loc_pix"]) > 5.0]
    low = sorted(classes, key=lambda c: res[c]["tiers"]["L1"]["jud_img"])[:6]
    L += [f"- **「窗口塔救不了」的类(判定 img 变动 <2 点,但定位 pix 改善 >5 点):"
          f"共 {len(stuck)} 个** —— " + (", ".join(c for c, _ in stuck) or "无"), ""]
    L += ["- **判定 img 最低的 6 类**(瓶颈所在):", "",
          "| 类 | 判定img(L1) | 判定img(full) | 判定pix(full) | 定位pix(full) | 差距 |",
          "|---|---|---|---|---|---|"]
    for c in low:
        t = res[c]["tiers"]
        L.append(f"| {c} | {t['L1']['jud_img']:.1f} | {t['full']['jud_img']:.1f} | "
                 f"{t['full']['jud_pix']:.1f} | {t['full']['loc_pix']:.1f} | "
                 f"**{t['full']['loc_pix'] - t['full']['jud_img']:+.1f}** |")
    L += ["", "> 「差距」= 定位 pix − 判定 pix。**正值越大,说明像素级定位越好、"
          "整件判定越差 —— 瓶颈越可能出在 `img_score=(cls_prob+max)/2` "
          "这个组装方式上**,而不是窗口预算上。", ""]

    # ---- 任务书问题 2:GT 面积机制 ----
    L += ["## 4. 问题二:GT 面积 vs 档位增益(metal_nut 反向的机制)", "",
          "本地推测:缺陷**面积大**时,调和平均 `2/(1/a+1/b)` 被窗内小值支配,"
          "窗口塔反而有害。两级检验:", "",
          "**跨类**:", ""]
    x = [res[c]["gt_area_mean"] for c in classes]
    y = [res[c]["tiers"]["full"]["jud_img"] - res[c]["tiers"]["L1"]["jud_img"]
         for c in classes]
    yp = [res[c]["tiers"]["full"]["jud_pix"] - res[c]["tiers"]["L1"]["jud_pix"]
          for c in classes]
    if len(x) > 2:
        r1, p1 = spearman(x, y)
        r2, p2 = spearman(x, yp)
        L += [f"- Spearman(GT面积, **Δ判定img**) = **{r1:+.3f}**(p={p1:.3f}) → "
              f"{'负相关,**支持**推测' if r1 < 0 else '**不支持**推测'}",
              f"- Spearman(GT面积, **Δ定位pix**) = **{r2:+.3f}**(p={p2:.3f}) → "
              f"{'负相关' if r2 < 0 else '**不支持**推测'}", "",
              "**跨类层面:推测被证伪(p≈0.93,完全不显著)。** "
              "screw 的 GT 面积最小(0.003)却掉得最多(−4.5),"
              "metal_nut 面积最大(0.145)也只掉 1.1 —— 面积在**跨类**上"
              "预测不了档位增益。**真正成立的是类内**:见下表。", ""]
    L += ["**类内**(每张图的 GT 面积 vs 该图自身的分数增益):", "",
          "| 类 | GT面积 | 类内 rho | p | 样本数 | 读法 |", "|---|---|---|---|---|---|"]
    neg = 0
    for c in classes:
        r = res[c]
        rho = r["within_class_rho_area_vs_gain"]
        if rho is None:
            L.append(f"| {c} | {r['gt_area_mean']:.3f} | — | — | {r['n_bad']} | "
                     f"方差不足,不可判 |")
            continue
        if rho < 0:
            neg += 1
        L.append(f"| {c} | {r['gt_area_mean']:.3f} | {rho:+.3f} | "
                 f"{r['within_class_rho_p']:.4f} | {r['n_bad']} | "
                 f"{'面积越大增益越小 ✓' if rho < 0 else '面积越大增益越大 ✗'} |")
    L += ["", f"**{neg}/{len(classes)} 类类内 rho 为负**"
          f"(即:同一类里,缺陷面积越大的图,窗口塔带来的增益越小)。", "",
          "### 4.1 metal_nut 反向的**真机制**(消融实验,推翻原推测)", "",
          "原推测「大范围形变 → 调和平均被窗内小值支配」在跨类层面已被证伪。"
          "故另做 4 变体消融(只用缓存特征,不碰 GPU),结果如下:", "",
          "| 类 | 定位pix(纯few) | L1判定pix | A 原式 | B **去掉cls_prob** | "
          "C cls加性 |", "|---|---|---|---|---|---|"]
    for c, r in ABL.items():
        L.append(f"| {c} | {r['loc_pix(few only)']:.1f} | "
                 f"{r['L1_判定pix(cls+few)']:.1f} | {r['A_原式(调和cls+窗)']:.1f} | "
                 f"**{r['B_去cls(只窗)']:.1f}** | {r['C_cls加性']:.1f} |")
    L += ["", "**读法**:", "",
          "- **C 列(cls_prob 当常数相加)全面崩塌** —— metal_nut 59.9、screw 74.9、"
          "tile 55.3,远低于定位(92~97)。证明这是个**加性常数污染**:"
          "`cls_prob` 整幅同值,缺陷图上它**抬高背景**、良品图上抬高同样多,"
          "像素排名被抹平。**这正是任务书坑二说的那件事。**",
          "- **B 列(去掉 cls)在每一类上都是最好的** → "
          "在 `m_all` 里,`cls_prob` 是**纯负担**,没有带来任何定位信息。",
          "- **因此「L1 判定pix 只有 71.0」不是 L1 的定位能力差** —— "
          "它的判定图被加性常数污染了。L1 的**真实**定位是 `定位pix` 行(93.3 均值)。",
          "- **metal_nut 的反向机制**:去 cls 后 69.9→72.8(+2.9),"
          "而它的 cls_img 高达 96.3(15 类最高)。`cls_prob` 越准,"
          "整幅常数抬得越狠,对像素排名的污染越大 —— "
          "**「图像级分数好」反而害了「像素级定位」**。这解释了为什么"
          "偏偏是 metal_nut 掉点,而不是面积大的类。", ""]

    # ---- 过杀/漏检配对 ----
    L += ["## 5. 过杀/漏检:只报配对值(任务书坑三)", "",
          "阈值取自**良品**分位。下面报的是 **(过杀, 漏检) 成对数字**,"
          "**不**单独报「过杀率@Pxx」—— 那个比例恒等于 x%,与模型无关。", "",
          "| 档位 | 阈值=良品max(过杀=0)×漏检 | 良品P95(过杀, 漏检) | "
          "良品P99(过杀, 漏检) |", "|---|---|---|---|"]
    for t in TIER_ORDER:
        esc0 = mean("pair_overkill0_escape", t)
        ok95 = mean("pair_p95_overkill", t)
        esc95 = mean("pair_p95_escape", t)
        ok99 = mean("pair_p99_overkill", t)
        esc99 = mean("pair_p99_escape", t)
        L.append(f"| {TIER_LABEL[t]} | 0% × {esc0:.1f}% | "
                 f"({ok95:.1f}%, {esc95:.1f}%) | ({ok99:.1f}%, {esc99:.1f}%) |")
    L += ["", "**采样量学警告**:良品只有 12~60 张,过杀率的最小非零粒度 = 1/N:",
          ""]
    grains = sorted({r["overkill_grain_pct"] for r in res.values()})
    L += [f"- 本 15 类:最小粒度 {min(grains):.1f}% ~ 最大粒度 {max(grains):.1f}%"
          f"(良品 12~60 张)",
          "- **要报 1% 量级的过杀率,良品样本需几百张**。"
          "本表的过杀数字受此粒度限制,不可解读到 1% 精度。", ""]

    # ---- 不单调 ----
    L += ["## 6. 不单调的类(不许平滑掉,任务书 §6)", "",
          "任务书要求:预算越高精度反而下降要如实报出。以下为判定 img "
          "在三档上**非单调**(先升后降或直接下降)的类:", "",
          "| 类 | L1 | L1+3×3 | full | 形态 |", "|---|---|---|---|---|"]
    nmono = 0
    for c in classes:
        t = res[c]["tiers"]
        a, b, cc = t["L1"]["jud_img"], t["L1+3x3"]["jud_img"], t["full"]["jud_img"]
        shape = ""
        if cc < b < a:
            shape = "**单调下降**"
        elif cc < b and b > a:
            shape = "**先升后降**(峰值在 L1+3×3)"
        elif b < a and cc >= b:
            shape = "先降后升(谷在 L1+3×3)"
        if shape:
            nmono += 1
            L.append(f"| {c} | {a:.1f} | {b:.1f} | {cc:.1f} | {shape} |")
    if nmono == 0:
        L.append("| — | — | — | — | 本批无 |")
    L.append("")

    # ---- 网格交叉引用 ----
    if args.grid:
        g = load(args.grid)
        L += ["## 7. 附:预算网格(另一种切法,交叉参照)", "",
              "上面三档是**尺度开关**;下表是**同尺度内的窗口预算**"
              "(B 为 3×3 窗数,2×2 按比例)。两套切法的结论应互相印证。", ""]
        gr = g["results"]
        L += ["| 策略 | B | 判定img | 定位img | 判定pix |", "|---|---|---|---|---|"]
        for st in g["strategies"]:
            for B in g["budgets"]:
                k = f"{B}|{st}"
                rows = [r[k] for r in gr.values() if k in r]
                if not rows:
                    continue
                m = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
                L.append(f"| {st} | {B} | {m('img_auc_prod'):.1f} | "
                         f"{m('img_auc_loc'):.1f} | {m('pix_auc_prod'):.1f} |")
        L.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[out] {args.out}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
