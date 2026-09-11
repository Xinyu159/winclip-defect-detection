"""任务 C:GPU 真实节拍表(P50/P95),按硬件分表。

任务书 T01 §4 的硬要求:
  - 单帧时延取 **P50 / P95**(产线看 P95,不看均值)
  - 每档预热 ≥10 帧、计时 ≥50 帧(在 bench_onnx.py 里保证)
  - **必须记录 GPU 型号与驱动**
  - **CPU 与 GPU 分表**,不许混

产出:`发件箱/T01_C_硬件时延表.md`

用法:
    python scripts/make_t01_C_table.py \
        --gpu logs/bench_taskC.json --cpu logs/bench_taskC_cpu.json \
        --out 发件箱/T01_C_硬件时延表.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ms(v: dict, key: str) -> str:
    x = v.get(key)
    return "—" if x is None else f"{x:.2f}"


def section(rep: dict, title: str, note: str) -> list[str]:
    hw = rep.get("hw", {})
    L = [f"## {title}", ""]
    if note:
        L += [note, ""]
    L += [f"- **GPU**:`{hw.get('gpu', '(未探测到)')}`",
          f"- **驱动**:见上行的 driver_version 字段",
          f"- **算力**:`{hw.get('compute_cap', '?')}`",
          f"- **ONNX Runtime**:`{hw.get('ort')}`",
          f"- **可用 EP**:`{', '.join(hw.get('ort_providers', []))}`",
          f"- 预热 {rep.get('n_warm')} 帧 / 计时 {rep.get('n_iter')} 次",
          f"- 测量时间:`{rep.get('ts')}`", ""]

    # 端到端整图(research 完整路径:patcher + 整图塔 + 两尺度全量窗口)
    L += ["### 端到端单帧整图(全量窗口 = research 完整路径)", "",
          "| 配置 | 实际生效 EP | 模型体积 | 类 | P50 (ms) | P95 (ms) | 均值 (ms) |",
          "|---|---|---|---|---|---|---|"]
    for c in rep.get("configs", []):
        if "error" in c:
            L.append(f"| {c['name']} | — | — | — | **未跑通**:{c['error']} | | |")
            continue
        size = c.get("model_sizes_mb", {})
        total = sum(size.values()) if size else 0
        for cls, v in c.get("per_class", {}).items():
            e = v.get("e2e_full", {})
            L.append(f"| {c['name']} | {c.get('active_provider')} | "
                     f"{total:.0f} MB | {cls} | {ms(e, 'p50_ms')} | "
                     f"**{ms(e, 'p95_ms')}** | {ms(e, 'mean_ms')} |")
    L.append("")

    # 窗口预算档位(级联实际形态)
    budgets = rep.get("budgets", [])
    if budgets:
        L += ["### 窗口预算档位(级联实际形态:只算 top-N 窗口的 L2 时延)", "",
              "| 配置 | 类 | " + " | ".join(f"B={b} P50" for b in budgets)
              + " |", "|---|---|" + "---|" * len(budgets)]
        for c in rep.get("configs", []):
            if "error" in c:
                continue
            for cls, per_b in c.get("budget_ms", {}).items():
                cells = [ms(per_b.get(str(b), {}), "p50_ms") for b in budgets]
                L.append(f"| {c['name']} | {cls} | " + " | ".join(cells) + " |")
        L += ["", "P95 明细:", "",
              "| 配置 | 类 | " + " | ".join(f"B={b} P95" for b in budgets)
              + " |", "|---|---|" + "---|" * len(budgets)]
        for c in rep.get("configs", []):
            if "error" in c:
                continue
            for cls, per_b in c.get("budget_ms", {}).items():
                cells = [ms(per_b.get(str(b), {}), "p95_ms") for b in budgets]
                L.append(f"| {c['name']} | {cls} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", default="")
    ap.add_argument("--contended", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu = json.loads(Path(args.gpu).read_text())
    L = ["# T01-C GPU 真实节拍(P50 / P95)", ""]
    L += ["**为什么本地数字不能用**:本地只有降频 CPU 的 5197ms,"
          "不代表目标硬件。产线节拍必须测于**目标硬件**。", ""]
    L += ["**测量纪律**(任务书 §4 硬要求):",
          "- 取 P50/P95,**不取均值** —— 产线看的是长尾,均值会把尖峰抹平",
          "- 每档预热 ≥10 帧、计时 ≥50 次",
          "- **测时 GPU 空闲**:与其它任务并发会把排队时间算进时延",
          "- CPU 与 GPU **分表**,不合成一张", ""]

    L += section(gpu, "一、GPU 表(目标硬件)", "")

    if args.contended:
        c = json.loads(Path(args.contended).read_text())
        L += ["## 一之二、为什么必须「测时 GPU 空闲」—— 本批的实测证据", "",
              "本任务第一次测量时,机器上另有一个 **CPU 满载**的 int8 评测作业"
              "在跑(GPU 利用率仅 20%,显存 684 MiB)。当时以为「GPU 没被占用」,"
              "于是直接测了一轮 —— **结果 P95 被抬高了 13 倍**:", "",
              "| 配置 | 档位 | 并发时 P95 | **空闲时 P95** | 倍数 |",
              "|---|---|---|---|---|"]
        for name, k, lbl in (("fp32cuda", "e2e_full", "e2e 全量"),
                             ("fp32cuda", "b8", "B=8"),
                             ("fp16cuda", "e2e_full", "e2e 全量"),
                             ("fp16cuda", "b8", "B=8")):
            def get(dd, nm, key):
                for cc in dd.get("configs", []):
                    if cc.get("name") != nm:
                        continue
                    if key == "e2e_full":
                        v = (cc.get("per_class", {}).get("tile", {})
                             .get("e2e_full", {}) or {})
                    else:
                        v = (cc.get("budget_ms", {}).get("tile", {})
                             .get("8", {}) or {})
                    return v.get("p95_ms")
                return None
            a = get(c, name, k)
            b = get(gpu, name, k)
            if a and b:
                L.append(f"| {name} | {lbl} | {a:.2f} ms | **{b:.2f} ms** | "
                         f"{a/b:.1f}× |")
        L += ["", "**注意**:并发作业是 **CPU** 密集的(int8 动态量化在 CUDA 上"
              "慢约 8 倍,实际跑在 CPU 上),GPU 利用率只有 20%、显存只占 684 MiB。"
              "**光看 GPU 利用率会误判为「空闲」**,但实测 P95 仍被抬高一个数量级 —— "
              "推测是主机侧调度与 ORT 线程争抢所致。", "",
              "→ **本节所有数字取「空闲时」那一列。并发那轮整体作废,"
              "不作为节拍数字引用。** 这也正是任务书 §4 写"
              "「测时 GPU 空闲」的实际价值。", ""]

    if args.cpu:
        cpu = json.loads(Path(args.cpu).read_text())
        L += section(cpu, "二、CPU 表(对照,不可与上表直接比较)",
                     "> ⚠ 本表与 GPU 表**不同硬件**,数字不可横向比较,"
                     "只能各自与自己比。")

    L += ["## 结论口径提醒", "",
          "本文件所有数字测于上表标注的**同一台机器**。"
          "与 `工作总览与行动清单.md` 里的本地 CPU 数字**不可直接比较** —— "
          "那是另一台硬件(降频 Intel),两者的比值不构成加速比证据。", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
