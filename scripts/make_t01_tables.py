"""把 cascade_v2 的网格 JSON 渲染成任务书要求的汇总表(全 15 类)。

任务书 T01 §2 指定三节,本脚本按序生成:
  表一  像素 AUROC(类 × 策略 × 预算)
  表二  img AUROC + 过杀@P99 + 漏检@P99(同一网格)
  表三  每类"能保住 full 精度 95% 的最小预算"及其过杀/漏检

另有"不单调类"清单(T01 §2 第 3 条硬要求):预算升高反而变差的条目必须
显式列出,不许平滑掉。

用法:
    python scripts/make_t01_tables.py --json logs/cascade_v2_*.json \
        --out 发件箱/T01_A_汇总表.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BUDGETS = [0, 8, 16, 32, 64, 128, 999]
STRATS = ["none", "cv", "fewshot", "rand"]

#: 表头用的短标签(999 是"全量"而非真实预算,数字直白写出容易被误读成预算)
BUDGET_LABEL = {999: "全量"}


def fmt(v, spec=".1f", dash="—"):
    return dash if v is None else format(v, spec)


def grid_table(res: dict, key: str, title: str, note: str = "") -> list[str]:
    """一个类 × 一个指标 → 一张 markdown 表。"""
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    head = "| 类 | 策略 | " + " | ".join(
        BUDGET_LABEL.get(b, f"B={b}") for b in BUDGETS) + " |"
    out += [head, "|" + "---|" * (len(BUDGETS) + 2)]
    for cls, rows in res.items():
        for st in STRATS:
            cells = []
            for b in BUDGETS:
                k = f"{b}|{st}"
                if k not in rows:
                    cells.append("·")          # 该组合不适用(B=0 只有 none)
                    continue
                cells.append(fmt(rows[k].get(key)))
            out.append(f"| {cls} | {st} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def min_budget_section(res: dict) -> list[str]:
    """每类:能保住 full(B=128,该策略下最优档)95% 精度的最小预算。

    口径写死在代码里而不是叙述里,避免解读时漂移:
      - 参考基准 = 同一策略、同一类、B=128 的 img AUROC(该策略自己的上界)
      - 达标 = img_auc >= 0.95 × 基准
      - 取满足条件的最小 B
    用"同策略自己的 B=128"而不是别的策略的数字做分母,是因为各策略的
    选窗依据不同,横向比较本就不公平;这里问的是"这个策略压到多小还能用"。
    """
    out = ["### 三、最小可用预算(保住同策略 B=128 的 95% img AUROC)", "",
           "| 类 | 策略 | B=128 img | 95% 线 | 最小预算 | 该档 img | 过杀@P99 | 漏检@P99 |",
           "|---|---|---|---|---|---|---|---|"]
    for cls, rows in res.items():
        for st in STRATS:
            top = rows.get(f"128|{st}")
            if not top or top.get("img_auc") is None:
                continue
            base = top["img_auc"]
            line = 0.95 * base
            hit = None
            for b in BUDGETS:
                r = rows.get(f"{b}|{st}")
                if r and r.get("img_auc") is not None and r["img_auc"] >= line:
                    hit = (b, r)
                    break
            if hit is None:
                out.append(f"| {cls} | {st} | {fmt(base)} | {fmt(line)} | "
                           f"**无一档达标** | — | — | — |")
            else:
                b, r = hit
                out.append(
                    f"| {cls} | {st} | {fmt(base)} | {fmt(line)} | **{b}** | "
                    f"{fmt(r.get('img_auc'))} | {fmt(r.get('overkill_p99'))}% | "
                    f"{fmt(r.get('escape_p99'))}% |")
    out.append("")
    return out


def nonmonotonic_section(res: dict, key: str) -> list[str]:
    """列出预算升高反而变差的条目 —— 真实现象,不许平滑掉。"""
    out = [f"### 不单调条目({key} 随预算升高反而下降)", ""]
    found = []
    for cls, rows in res.items():
        for st in STRATS:
            prev, prev_b = None, None
            for b in BUDGETS:
                r = rows.get(f"{b}|{st}")
                if not r or r.get(key) is None:
                    continue
                v = r[key]
                if prev is not None and v < prev - 1e-9:
                    found.append((cls, st, prev_b, prev, b, v))
                prev, prev_b = v, b
    if not found:
        out += ["(无)", ""]
        return out
    out += ["| 类 | 策略 | 从 | 到 | 变化 |", "|---|---|---|---|---|"]
    for cls, st, b0, v0, b1, v1 in found:
        out.append(f"| {cls} | {st} | B={b0} {v0:.1f} | B={b1} {v1:.1f} | "
                   f"{v1 - v0:+.1f} |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="T01-A 级联全量网格扫描")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    meta, res = data["meta"], data["results"]

    L = [f"# {args.title}", ""]
    L += ["## 0. 测量条件(数字脱离这些没有意义)", "",
          f"- 后端/设备:`{meta.get('backend')}` / `{meta.get('device')}`",
          f"- 缺陷缓存:`{meta.get('cache')}`",
          f"- 良品缓存:`{meta.get('cache_good')}`(阈值来源,467 张)",
          f"- 生成时间:`{meta.get('ts')}`",
          f"- 类:{', '.join(meta.get('classes', []))}",
          f"- 预算档:{meta.get('budgets')}",
          f"- 选窗策略:{', '.join(meta.get('strategies', []))}",
          ""]
    L += ["**策略含义**:`none` = 不精检(B=0 纯地基,无选窗预算);"
          "`cv` = L0 传统 CV 可疑度;`fewshot` = L1 已算的 patch few 分;"
          "`rand` = 随机(下界对照)。三者均**不使用 GT**。", ""]
    L += ["**阈值口径**:过杀率/漏检率的分位阈值**只从良品 img_score 标定**"
          "(P99 → 99 分位)。缺陷品不参与定阈。", ""]

    L += grid_table(res, "pix_auc", "一、像素 AUROC(辅助指标)",
                    "低预算下该指标会失真:未被精检的区域保留地基分,"
                    "逐像素排名会把它们当成\"判为正常\",故**不作为选型主依据**。")
    L += grid_table(res, "img_auc", "二、img AUROC(主指标,产线口径)")
    L += grid_table(res, "overkill_p99", "二-A、过杀率 @P99(%)—— 良品被误判为 NG 的比例")
    L += grid_table(res, "escape_p99", "二-B、漏检率 @P99(%)—— 缺陷品被漏判的比例")
    L += min_budget_section(res)
    L += nonmonotonic_section(res, "img_auc")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[out] {args.out}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
