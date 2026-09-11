"""产线标定:成本模型(节拍) + 良品阈值(判定) + 定位模板。

三件事都在这一步做完,因为它们是同一件事的三个面:**上线前用良品把
"机器有多快"和"什么算异常"同时定下来**。运行期只读标定产物,不再碰参数。

严格约束(与 research 评估协议一致):
    阈值只能来自良品类(Nd / train/good)。绝不允许用缺陷图或 GT 反推
    —— 否则换一批货阈值立刻失效,且离线指标会虚假偏高。

用法:
    # 1) 成本模型(换工控机后必跑,一条命令,不需要数据集)
    python scripts/line_calib.py cost --deploy data/deploy --out data/deploy/cost_model.json

    # 2) 良品阈值 + 定位模板
    python scripts/line_calib.py station --config configs/line_prod_4i.yaml \
        --good_dir /path/to/Nd --out data/deploy/guard_Steel_Sc.json

    # 3) 只标定位模板(全幅面模式需要)
    python scripts/line_calib.py locate --good_dir ... --out data/deploy/template.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from runtime.line.config import load_config                # noqa: E402
from runtime.line.localizer import WorkpieceLocalizer      # noqa: E402
from runtime.line.preprocess_cv import (                   # noqa: E402
    build_flatfield, normalize_illumination)
from runtime.line.scheduler import GRID, calibrate         # noqa: E402
from runtime.line.verdict import calibrate_from_good       # noqa: E402
from runtime.ov_engine import OVEngine                     # noqa: E402
from runtime.pipeline import OVPipeline                    # noqa: E402


def read_gray_dir(d: str | Path, limit: int = 0) -> list[tuple[str, np.ndarray]]:
    root = Path(d)
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if limit:
        files = files[:limit]
    out = []
    for p in files:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            out.append((p.name, g))
    return out


# ----------------------------------------------------------------------
def cmd_cost(args) -> int:
    print("[cost] 标定线性成本模型 ms = fixed + tokens × slope", flush=True)
    engine = OVEngine(args.deploy, args.device)
    t0 = time.time()
    cm = calibrate(engine, n_repeat=args.repeat)
    print(f"[cost] ms_per_token={cm.ms_per_token:.4f}  fixed_ms={cm.fixed_ms:.2f}"
          f"  ({cm.source})  {time.time()-t0:.0f}s", flush=True)

    # 外推校验:全量窗口的预测值 vs 预测公式
    pred_full = cm.estimate_ms(169, 196, use_full=True)
    print(f"[cost] 外推校验:全量窗口预计 {pred_full:.0f} ms"
          f"(若机器降频,batch 1→169 会偏离线性,见 README 说明)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ms_per_token": cm.ms_per_token, "fixed_ms": cm.fixed_ms,
        "source": cm.source,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "换工控机/换 OpenVINO 版本后重跑本命令",
    }, ensure_ascii=False, indent=2))
    print(f"[cost] → {out}")
    return 0


def cmd_locate(args) -> int:
    cfg = load_config(args.config) if args.config else None
    from runtime.line.config import CvCfg, GeometryCfg
    gcfg = GeometryCfg(mode="full_frame", template_path=str(args.out),
                       search_px=args.search_px)
    ccfg = cfg.cv if cfg else CvCfg()
    items = read_gray_dir(args.good_dir, args.limit)
    if not items:
        print(f"[locate] 良品目录为空: {args.good_dir}")
        return 1
    loc = WorkpieceLocalizer.from_normal_frames(
        [g for _, g in items], gcfg, ccfg)
    loc.save(args.out)
    print(f"[locate] {len(items)} 帧 → 模板 + 外接框 {loc.bbox} → {args.out}")
    return 0


def cmd_station(args) -> int:
    """良品阈值 + 平场(可选)+ 定位模板(可选)。"""
    cfg = load_config(args.config)
    if cfg.cascade.early_exit_guard_path:
        print("[warn] 配置里已有 guard 路径;本次标定会覆盖它")

    items = read_gray_dir(args.good_dir, args.limit)
    if len(items) < 20:
        print(f"[station] 良品样本仅 {len(items)} 张,标定分位数不稳"
              f"(建议 ≥50 张,产线实测 200+ 更稳)")
        if len(items) == 0:
            return 1
    print(f"[station] 良品 {len(items)} 张 | 类名 {cfg.class_name}", flush=True)

    # ---- 平场(可选,推荐)------------------------------------------
    if cfg.cv.illum_norm == "flatfield" and args.save_flatfield:
        ff = build_flatfield([g for _, g in items])
        np.savez(args.save_flatfield, flatfield=ff)
        print(f"[station] 平场参考 → {args.save_flatfield}")

    # ---- 定位模板(全幅面模式)-------------------------------------
    if cfg.geometry.mode == "full_frame" and args.save_template:
        loc = WorkpieceLocalizer.from_normal_frames(
            [g for _, g in items], cfg.geometry, cfg.cv)
        loc.save(args.save_template)
        print(f"[station] 定位模板 → {args.save_template} | bbox={loc.bbox}")

    # ---- 逐帧跑模型收集良品分布 ------------------------------------
    engine = OVEngine(cfg.deploy_dir, cfg.device)
    pipe = OVPipeline(engine, cfg.text_dir)
    pipe.set_class(cfg.class_name)
    if cfg.cascade.use_few and args.gallery_dir:
        gs = read_gray_dir(args.gallery_dir, args.shots)
        if gs:
            imgs = np.concatenate([engine.preprocess_rgb(
                _to_rgb(g, cfg)) for _, g in gs], axis=0)
            pipe.set_gallery(imgs)
            print(f"[station] gallery {len(gs)} 张(影响 few 分支的分数尺度)")

    flat = None
    if cfg.cv.illum_norm == "flatfield":
        p = Path(cfg.cv.flatfield_path)
        if p.exists():
            flat = np.load(p)["flatfield"].astype(np.float32)

    from runtime.line.localizer import build_roi_weight
    from runtime.line.preprocess_cv import patch_suspicion, suspicion_map
    roi_w = build_roi_weight(cfg.roi, 240)
    loc_obj = None
    if cfg.geometry.mode == "full_frame" and args.save_template:
        loc_obj = WorkpieceLocalizer.from_file(
            args.save_template, cfg.geometry, cfg.cv)

    scores, patch_max = [], []
    t0 = time.time()
    for i, (name, g) in enumerate(items):
        std = g
        if loc_obj is not None:
            lr = loc_obj.locate(g)
            std = loc_obj.workpiece_crop(loc_obj.warp_to_standard(g, lr))
        norm = normalize_illumination(std, cfg.cv, flat)
        x = _to_model_input(norm)
        toks = engine.patcher(x)
        full = engine.tower_full(toks)
        cp = float(pipe._prob(full[0, :1], pipe.pos, pipe.neg, pipe.temp)[0])
        pp = pipe._prob(full[0, 1:], pipe.pos, pipe.neg, pipe.temp)
        scores.append(cp); patch_max.append(float(pp.max()))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(items)}] {time.time()-t0:.0f}s "
                  f"img={cp:.4f} patchmax={pp.max():.4f}", flush=True)

    guard = calibrate_from_good(scores, patch_max,
                                quantile=args.patch_quantile,
                                target_fpr=args.target_fpr)
    guard.update({"class_name": cfg.class_name,
                  "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "illum_norm": cfg.cv.illum_norm,
                  "n_good": len(items),
                  "source": str(args.good_dir)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(guard, ensure_ascii=False, indent=2))

    print(f"\n[station] 良品分布 image: mean={guard['mean_image_score']:.4f} "
          f"std={guard['std_image_score']:.4f}")
    print(f"[station] 建议阈值(image_threshold={guard['image_threshold']:.4f} "
          f"= 良品 {1-args.target_fpr:.1%} 分位,目标过杀率 {args.target_fpr:.1%})")
    print(f"[station] 热力图阈值(map_threshold={guard['map_threshold']:.4f} "
          f"= 良品 patch 峰值 {args.patch_quantile:.1%} 分位)")
    print(f"[station] 早退护栏 image_p999={guard['image_p999']:.4f} "
          f"patch_p999={guard['patch_p999']:.4f}")
    print(f"[station] → {out}\n")
    return 0


def _to_rgb(g: np.ndarray, cfg) -> np.ndarray:
    from runtime.line.preprocess_cv import normalize_illumination
    n = normalize_illumination(g, cfg.cv)
    n = cv2.resize(n, (240, 240), interpolation=cv2.INTER_LINEAR)
    return np.clip(n, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)


def _to_model_input(norm: np.ndarray) -> np.ndarray:
    g = norm.astype(np.float32)
    if g.shape != (240, 240):
        g = cv2.resize(g, (240, 240), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(g, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    return OVEngine.preprocess_rgb(rgb)


def main() -> int:
    ap = argparse.ArgumentParser(description="产线标定:成本模型 / 良品阈值 / 定位模板")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cost", help="标定节拍成本模型")
    c.add_argument("--deploy", default="data/deploy")
    c.add_argument("--device", default="CPU")
    c.add_argument("--repeat", type=int, default=3)
    c.add_argument("--out", default="data/deploy/cost_model.json")
    c.set_defaults(func=cmd_cost)

    l = sub.add_parser("locate", help="标定定位模板(全幅面产线)")
    l.add_argument("--good_dir", required=True)
    l.add_argument("--config", default="")
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--search_px", type=int, default=40)
    l.add_argument("--out", default="data/deploy/template.npz")
    l.set_defaults(func=cmd_locate)

    s = sub.add_parser("station", help="标定良品阈值(过杀率口径)")
    s.add_argument("--config", required=True)
    s.add_argument("--good_dir", required=True)
    s.add_argument("--gallery_dir", default="")
    s.add_argument("--shots", type=int, default=4)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--target_fpr", type=float, default=0.01,
                   help="目标过杀率(良品分位口径)")
    s.add_argument("--patch_quantile", type=float, default=0.999)
    s.add_argument("--save_flatfield", default="")
    s.add_argument("--save_template", default="")
    s.add_argument("--out", default="data/deploy/guard.json")
    s.set_defaults(func=cmd_station)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
