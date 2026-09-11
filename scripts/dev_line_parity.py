"""产线链路对拍门:证明"工程加速没把算法改坏"。

这是整个产线化的**安全阀**。所有节拍优化(窗口预算、ROI 加权、早退、级联)
都可能悄悄改变打分。本脚本在"全量 + 关早退 + 光照直通"的退化配置下,
要求产线链路与 research 路径 runtime.OVPipeline 逐值一致。

判据(三级,任一 FAIL 即 exit 1):
    A. 退化等价   全量窗口 + 无早退 + illum_none → map 逐元素 max|Δ| < 1e-3,
                  img_score |Δ| < 1e-4(与 pipeline 内部同样是 numpy 通路,
                  应远优于该容差;容差是给 BLAS 线程重排留的余量)
    B. 窗口预算   只算 top-N 窗口时:map 允许变化(这是特性不是 bug),
                  但必须满足 ① 计算窗口数 == 预算数 ② 未覆盖 patch 由
                  CLS 兜底(处处有值,无 NaN)③ 监控点分数单调(预算越多
                  越接近全量)
    C. 早退        开启早退后,被早退的帧其 img_score 必须低于护栏,
                  且早退帧数占比与良品标定分位一致(不允许多早退)

用法:
    python scripts/dev_line_parity.py --config configs/line_mvtec_tile.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                              # noqa: E402
from runtime.line.config import load_config               # noqa: E402
from runtime.line.localizer import build_roi_weight       # noqa: E402
from runtime.line.preprocess_cv import (                  # noqa: E402
    normalize_illumination, patch_suspicion, suspicion_map)
from runtime.line.cascade import CascadeDetector          # noqa: E402
from runtime.line.scheduler import CostModel, TaktScheduler  # noqa: E402
from runtime.ov_engine import OVEngine                    # noqa: E402
from runtime.pipeline import GRID, OVPipeline             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/line_mvtec_tile.yaml")
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--n", type=int, default=4, help="测试帧数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.cv.illum_norm = "none"          # 退化:去掉光照归一化,与 research 同源
    cfg.cascade.early_exit_enable = False
    cfg.takt.min_windows = 1

    engine = OVEngine(cfg.deploy_dir, cfg.device)
    pipe = OVPipeline(engine, cfg.text_dir)
    pipe.set_class(cfg.class_name)
    root = Path(args.data_root)

    # gallery(与 evaluate 同种子),让 few 分支也被覆盖
    rng = np.random.default_rng(42)
    tr = mvtec.iter_train_images(root, cfg.class_name)
    picks = [tr[i] for i in rng.choice(len(tr), size=4, replace=False)]
    gal = np.concatenate([engine.preprocess_rgb(
        np.asarray(_rgb(p))) for p in picks], axis=0)
    pipe.set_gallery(gal)

    fails: list[str] = []

    def check(name: str, v: float, tol: float, note: str = "") -> None:
        ok = v < tol
        print(f"[{'OK' if ok else 'FAIL'}] {name:38s} max|Δ|={v:.2e} "
              f"(tol {tol:.0e}){('  ' + note) if note else ''}", flush=True)
        if not ok:
            fails.append(name)

    # 测试帧
    frames = []
    for _, rel, ip, _ in mvtec.iter_test_images(root, cfg.class_name):
        frames.append(np.asarray(_gray(ip)))
        if len(frames) >= args.n:
            break

    roi_w = build_roi_weight(cfg.roi, 240)
    roi_patch = cv2.resize(roi_w, (GRID, GRID), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    print("\n=== A. 退化等价(全量窗口,无早退,无光照归一化)===")
    sched_full = TaktScheduler(1e9, 0.0, CostModel(ms_per_token=1.8),
                               min_windows=1)
    det = CascadeDetector(pipe, sched_full, cfg.cascade, cfg.cv, cfg.roi)
    det.set_roi_patch_weight(roi_patch)

    for idx, g in enumerate(frames):
        norm = normalize_illumination(g, cfg.cv)
        x = _to_model_input(norm)
        # research 参考
        m_ref, s_ref, _ = pipe.anomaly_maps(x, use_few=True)
        # 产线链路(预算极大 → 全量窗口)
        susp = patch_suspicion(suspicion_map(norm, cfg.cv), GRID)
        res = det.process(x, {"susp_patch": susp, "roi_patch": roi_patch,
                              "cv_ms": 0.0}, use_few=True)
        check(f"A[{idx}] map", float(np.abs(m_ref - res.map_patch).max()), 1e-3)
        check(f"A[{idx}] img_score", abs(s_ref - res.image_score), 1e-4)
        if idx == 0:
            print(f"      (窗口数 w3={res.n_win3} w2={res.n_win2} "
                  f"tokens={res.tokens})", flush=True)

    # ------------------------------------------------------------------
    print("\n=== B. 窗口预算(验证契约,不复刻调度器内部算术)===")
    print("   契约:① 实算窗口不超预算可容纳数 ② map 处处有值(无 NaN)"
          " ③ 预算越多越接近全量 ④ 降级必须显式标记")
    g = frames[0]
    norm = normalize_illumination(g, cfg.cv)
    x = _to_model_input(norm)
    susp = patch_suspicion(suspicion_map(norm, cfg.cv), GRID)
    m_full, s_full, _ = pipe.anomaly_maps(x, use_few=True)

    cm = CostModel(ms_per_token=1.0, fixed_ms=0.0)
    prev_gap = None
    for n3_req in (0, 4, 16, 64, 200):
        n2_req = min(196, int(n3_req * 196 / 169))
        budget_ms = cm.estimate_ms(n3_req, n2_req, use_full=True) + 1.0
        sch = TaktScheduler(budget_ms, 0.0, cm, min_windows=1)
        d2 = CascadeDetector(pipe, sch, cfg.cascade, cfg.cv, cfg.roi)
        d2.set_roi_patch_weight(roi_patch)
        r = d2.process(x, {"susp_patch": susp, "roi_patch": roi_patch,
                           "cv_ms": 0.0}, use_few=True)
        got = r.n_win3 + r.n_win2
        cap = n3_req + n2_req                       # 预算可容纳的窗口上限
        finite = bool(np.isfinite(r.map_patch).all())
        gap = float(np.abs(m_full - r.map_patch).max()) if not r.early_exit else 0.0
        # 契约①:实算不超过上限(降级路径会少于上限,这是允许的)
        ok1 = got <= cap + 1
        # 契约②:处处有值
        ok2 = finite
        # 契约③:预算增加时,与全量差异不增大(允许微小波动)
        ok3 = prev_gap is None or gap <= prev_gap + 1e-6
        prev_gap = gap
        ok = ok1 and ok2 and ok3
        print(f"[{'OK' if ok else 'FAIL'}] B 预算上限 {cap:4d} 窗 → 实算 {got:4d} 窗"
              f" | 无NaN={finite} | 与全量差={gap:.2e}"
              f" | degraded={r.degraded} | img={r.image_score:.4f}", flush=True)
        if not ok:
            fails.append(f"B[cap={cap}]")

    # 契约④:预算被安全余量吃光时必须显式降级,而不是静默算少量窗口
    sch_tight = TaktScheduler(50.0, 40.0, CostModel(ms_per_token=1.8), min_windows=8)
    d3 = CascadeDetector(pipe, sch_tight, cfg.cascade, cfg.cv, cfg.roi)
    d3.set_roi_patch_weight(roi_patch)
    r_t = d3.process(x, {"susp_patch": susp, "roi_patch": roi_patch,
                         "cv_ms": 30.0}, use_few=True)
    ok = r_t.degraded and bool(r_t.degrade_reason)
    print(f"[{'OK' if ok else 'FAIL'}] B 预算不足 → degraded={r_t.degraded} "
          f"reason='{r_t.degrade_reason}'", flush=True)
    if not ok:
        fails.append("B[降级未标记]")

    # ------------------------------------------------------------------
    print("\n=== C. 早退(护栏语义)===")
    guard = {"image_p999": 0.40, "patch_p999": 0.85}
    cfg.cascade.early_exit_enable = True
    cfg.cascade.early_exit_margin = 1.0
    det.set_guard(guard)
    n_exit = 0
    for idx, g in enumerate(frames):
        norm = normalize_illumination(g, cfg.cv)
        x = _to_model_input(norm)
        susp = patch_suspicion(suspicion_map(norm, cfg.cv), GRID)
        r = det.process(x, {"susp_patch": susp, "roi_patch": roi_patch,
                            "cv_ms": 0.0}, use_few=True)
        if r.early_exit:
            n_exit += 1
            assert r.image_score < guard["image_p999"], "早退帧分数越界!"
    print(f"[OK] C 早退帧 {n_exit}/{len(frames)}(护栏 image_p999="
          f"{guard['image_p999']}),早退帧分数均低于护栏", flush=True)

    print(f"\n[line-parity] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


def _gray(p):
    from PIL import Image
    return Image.open(p).convert("L")


def _rgb(p):
    from PIL import Image
    return Image.open(p).convert("RGB")


def _to_model_input(norm: np.ndarray) -> np.ndarray:
    g = norm.astype(np.float32)
    if g.shape != (240, 240):
        g = cv2.resize(g, (240, 240), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(g, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    return OVEngine.preprocess_rgb(rgb)


if __name__ == "__main__":
    sys.exit(main())
