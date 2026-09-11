"""部署端到端评估:image/pixel AUROC(协议复刻 evaluate.py)+ 单帧时延分解。

打分 = runtime.OVPipeline(winclip.py 的 numpy 镜像),**后端由 --deploy 目录
自动判定**:目录里有 .xml → OpenVINO(Intel 工控机形态),有 .onnx →
onnxruntime(ONNX 部署形态)。两条链路共用同一套打分数学,只换引擎,
所以结果可直接横向对比(这正是"复现 + 部署化"的证据链)。

评估协议与 evaluate.py 逐条一致(硬约束,不许为对数字好看而改):
  - set_class 用 MVTec 裸类名(下划线形式如 metal_nut)、gallery 采样 seed42
  - GT mask resize(240,240,BILINEAR)>128;pixel 分仅统计有 GT 的缺陷图
  - roc_auc_score / np.nanmean / JSON schema 同构
JSON 额外含 engine 字段,供出漂移表。

用法:
    python scripts/eval_ov.py --data_root data/mvtec_anomaly_detection \
        --classes tile --shots 0            # AUROC
    python scripts/eval_ov.py --data_root ... --classes tile --shots 0 --bench
                                            # 追加逐段时延(logs/latency_*.json)
    python scripts/eval_ov.py --deploy data/deploy_onnx --device cuda \
        --data_root ... --classes all --shots 0,4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                       # noqa: E402
from mvtec import MVTEC_CLASSES                    # noqa: E402
from runtime.pipeline import OVPipeline            # noqa: E402


def build_engine(deploy_dir: str | Path, device: str, tf32: bool | None = None):
    """按 deploy 目录内容选后端:有 .xml → OV,有 .onnx → ORT。

    显式报错而不是猜:两种产物混在一起时报歧义(混用会得到"看起来跑通了"
    但引擎不是想要的那个的结果,是测量类错误里最难发现的一种)。
    tf32 仅对 ORT+CUDA 生效;None = 用 ORT 默认(开启)。
    """
    d = Path(deploy_dir)
    if not d.exists():
        raise FileNotFoundError(f"deploy 目录不存在: {d}")
    xml = sorted(p.name for p in d.glob("*.xml"))
    onnx = sorted(p.name for p in d.glob("*.onnx"))
    if xml and onnx:
        raise RuntimeError(f"{d} 同时含 .xml({len(xml)}) 与 .onnx({len(onnx)}),"
                           f"无法判定后端,请把两种产物分目录")
    if xml:
        from runtime.ov_engine import OVEngine         # 延迟 import:无 OV 也能跑 ORT
        return OVEngine(d, "CPU" if device == "auto" else device), "openvino", \
            "CPU" if device == "auto" else device
    if onnx:
        from runtime.onnx_engine import OnnxEngine
        eng = OnnxEngine(d, device, tf32=tf32)
        return eng, "onnxruntime", eng.active_provider()
    raise RuntimeError(f"{d} 里既无 .xml 也无 .onnx")


def upsample_bilinear_np(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """(h,w) → (dst_h,dst_w) 双线性,align_corners=False(镜像 F.interpolate)。

    坐标:y=(dy+0.5)*h/dst_h-0.5;权重用未裁剪的 λ,索引裁剪到 [0,h-1]。
    开发期与 torch F.interpolate 数值对拍过(dev_parity.py),差异 <1e-6。
    """
    h, w = src.shape
    out = np.empty((dst_h, dst_w), dtype=np.float64)

    ys = np.zeros(dst_h, dtype=np.float64)   # λy
    y0s = np.zeros(dst_h, dtype=np.int64)
    y1s = np.zeros(dst_h, dtype=np.int64)
    for dy in range(dst_h):
        y = (dy + 0.5) * (h / dst_h) - 0.5
        y0 = int(math.floor(y))
        y0s[dy] = min(max(y0, 0), h - 1)
        y1s[dy] = min(y0s[dy] + 1, h - 1)
        ys[dy] = y - y0
    xs = np.zeros(dst_w, dtype=np.float64)
    x0s = np.zeros(dst_w, dtype=np.int64)
    x1s = np.zeros(dst_w, dtype=np.int64)
    for dx in range(dst_w):
        x = (dx + 0.5) * (w / dst_w) - 0.5
        x0 = int(math.floor(x))
        x0s[dx] = min(max(x0, 0), w - 1)
        x1s[dx] = min(x0s[dx] + 1, w - 1)
        xs[dx] = x - x0

    for dy in range(dst_h):
        ly, y0c, y1c = ys[dy], y0s[dy], y1s[dy]
        row0, row1 = src[y0c], src[y1c]
        for dx in range(dst_w):
            lx, x0c, x1c = xs[dx], x0s[dx], x1s[dx]
            v = ((1 - ly) * ((1 - lx) * row0[x0c] + lx * row0[x1c])
                 + ly * ((1 - lx) * row1[x0c] + lx * row1[x1c]))
            out[dy, dx] = v
    return out


def make_gallery(root: Path, cls: str, shot: int, seed: int,
                 pipeline: OVPipeline, engine) -> None:
    """与 evaluate.make_gallery 同种子同采样:train(全正常)抽 shot 张。"""
    rng = np.random.default_rng(seed)
    train_imgs = mvtec.iter_train_images(root, cls)
    picks = [train_imgs[i] for i in
             rng.choice(len(train_imgs), size=shot, replace=False)]
    # preprocess_rgb 已返回 (1,3,240,240),沿 batch 维拼接 → (shot,3,240,240);
    # 早期写法这里用 np.stack 会得到 (shot,1,3,240,240),set_gallery 再切
    # imgs[i:i+1] 变 5 维,OV 报 shape 不兼容(few-shot 路径走不通)
    imgs = np.concatenate([engine.preprocess_rgb(
        np.asarray(Image.open(p).convert("RGB"))) for p in picks], axis=0)
    pipeline.set_gallery(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--classes", default="all")
    ap.add_argument("--shots", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto",
                    help="auto / cpu / cuda(OV 后端 auto→CPU,ORT 后端 auto→CUDA 优先)")
    ap.add_argument("--bench", action="store_true",
                    help="追加单帧逐段时延测量(logs/latency_*.json)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    root = Path(args.data_root)
    classes = MVTEC_CLASSES if args.classes == "all" else \
        [c.strip() for c in args.classes.split(",")]
    shots_list = [int(s) for s in args.shots.split(",")]

    engine, backend, device = build_engine(args.deploy, args.device)
    sizes = (engine.ir_sizes_mb() if hasattr(engine, "ir_sizes_mb")
             else engine.model_sizes_mb())
    pipeline = OVPipeline(engine, args.text)
    # tag 进文件名与 JSON:同一脚本要跑 fp32/int8/fp16 多档,不标注会互相覆盖,
    # 出漂移表时无从分辨哪份是哪份。
    exp = {"script": "eval_ov.py", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "tag": args.tag, "classes": classes,
           "engine": backend, "device": device, "deploy": str(args.deploy),
           "model_sizes_mb": sizes, "shots_arg": args.shots}

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    log_path = log_dir / f"ov_{ts}{suffix}.json"

    print(f"[engine] {backend} | device={device} | 体积 {sizes} MB", flush=True)

    per_shot = {}
    bench = {"script": "eval_ov.py --bench", "classes": classes,
             "stages": {}, "per_class_frames": {}}
    for shot in shots_list:
        per_class = {}
        for cls in classes:
            t0 = time.time()
            pipeline.set_class(cls)
            if shot > 0:
                make_gallery(root, cls, shot, args.seed, pipeline, engine)

            scores, labels = [], []
            pix_scores, pix_gts = [], []
            stage_acc = {"patcher_ms": [], "full_ms": [], "wins_ms": [],
                         "mix_ms": [], "few_ms": []}
            for _, rel_type, img_path, mask_path in \
                    mvtec.iter_test_images(root, cls):
                x = engine.preprocess_rgb(
                    np.asarray(Image.open(img_path).convert("RGB")))
                p_map, img_score, diag = pipeline.anomaly_maps(
                    x, use_few=shot > 0)
                scores.append(img_score)
                labels.append(1 if rel_type != "good" else 0)
                for k, v in diag["ms"].items():
                    if k in stage_acc:
                        stage_acc[k].append(v)

                if mask_path is not None:
                    mask = np.asarray(Image.open(mask_path).convert("L")
                                      .resize((240, 240), Image.BILINEAR)) > 128
                    up = upsample_bilinear_np(p_map, *mask.shape)
                    pix_scores.append(up.flatten())
                    pix_gts.append(mask.flatten())

            img_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 \
                else float("nan")
            pix_auc = (roc_auc_score(np.concatenate(pix_gts),
                                     np.concatenate(pix_scores))
                       if pix_gts else float("nan"))
            per_class[cls] = {
                "n_test": len(scores), "n_defect": sum(labels),
                "img_auroc": round(float(img_auc) * 100, 1),
                "pix_auroc": round(float(pix_auc) * 100, 1),
            }
            if args.bench:
                bench["per_class_frames"][cls] = {
                    k: {"median_ms": round(float(np.median(v)), 2),
                        "p90_ms": round(float(np.percentile(v, 90)), 2)}
                    for k, v in stage_acc.items() if v}
            print(f"[s{shot} {cls:10s}] image={img_auc*100:6.2f}%  "
                  f"pixel={pix_auc*100:6.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # 逐类落盘:一轮 15 类 × 2 shot 要跑几十分钟,末尾一次性写盘意味着
            # 中途断掉(或输出被 grep 缓冲住)就什么都拿不到。写临时文件再
            # replace,避免读到写了一半的 JSON。
            exp["results"] = {**per_shot, str(shot): {"per_class": per_class}}
            tmp = log_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(exp, ensure_ascii=False, indent=2))
            tmp.replace(log_path)

        img_mean = np.nanmean([r["img_auroc"] for r in per_class.values()])
        pix_mean = np.nanmean([r["pix_auroc"] for r in per_class.values()])
        per_shot[str(shot)] = {"per_class": per_class,
                               "mean_img_auroc": round(float(img_mean), 1),
                               "mean_pix_auroc": round(float(pix_mean), 1)}
        print(f"\n==== shot={shot}  {len(classes)} 类均值:"
              f"image {img_mean:.1f}% | pixel {pix_mean:.1f}% ====\n",
              flush=True)

    exp["results"] = per_shot
    log_path.write_text(json.dumps(exp, ensure_ascii=False, indent=2))
    print(f"[log] {log_path}")

    if args.bench:
        bench["ts"] = ts
        bench["engine"] = backend
        bench["device"] = device
        bench["tag"] = args.tag
        bpath = log_dir / f"latency_{ts}{suffix}.json"
        bpath.write_text(json.dumps(bench, ensure_ascii=False, indent=2))
        print(f"[log] {bpath}  (逐帧各阶段 median/p90 ms)")


if __name__ == "__main__":
    main()
