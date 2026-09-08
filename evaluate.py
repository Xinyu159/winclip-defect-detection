"""WinCLIP v2 评估:image / pixel AUROC(逐类 + 15 类均值),zero-/few-shot。

对齐论文公式(官方 main.py + 社区复现 zqhang):
  - image 分数 = 整图 CLS 特征 ×(CLIP 温度 100)× softmax 的异常类概率
  - zero pixel map = 多尺度窗口异常概率(2×2 / 3×3)三路调和 → 15×15 → 上采样 GT
  - few-shot(全权重冻结):k 张正常参考图的三尺度特征作 gallery,
    查询窗口/patch 最近邻 0.5·(1−cos);few map = 三尺度平均,map += few map,
    image = (文本概率 + max(few map))/2
MVTec 图像全为方形(512/1024),preprocess 统一到 240,GT mask 同步 resize,
网格对齐。

用法示例:
    python evaluate.py --data_root /root/autodl-tmp/mvtec_anomaly_detection \\
        --weights /root/autodl-tmp/winclip_official/vit_b_16_plus_240-laion400m_e31-8fb26589.pt
    python evaluate.py --data_root ... --shots 1 --classes bottle,carpet --seed 42

实验参数与指标全量写入 logs/exp_<时间戳>.json,参数不进文件名。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# cuBLAS workspace 固定 + 确定性内核:消除启动期 GEMM heuristic 的跨进程
# 抖动(实测:TF32 关闭后同代码两次进程 img AUROC 仍可差 ~2pt,逐位不可复现)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

import mvtec
from mvtec import MVTEC_CLASSES
from winclip import WinCLIP

# 关闭 TF32 + 强制确定性内核(见上方注释)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="mvtec_anomaly_detection 目录")
    ap.add_argument("--classes", default="all",
                    help="逗号分隔类别;默认 all(15 类)")
    ap.add_argument("--shots", default="0",
                    help="逗号分隔的 few-shot 正常参考数;0=zero-shot")
    ap.add_argument("--seed", type=int, default=42,
                    help="参考图采样种子(固定后逐位可复现)")
    ap.add_argument("--weights", default="laion400m_e31",
                    help="ckpt 文件路径或 hub 预训练标签;勿留空(空=随机初始化)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="实验备注,写入 log")
    return ap.parse_args()


def up_to_gt(p_map: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """(1,1,h,w) anomaly map → bilinear 上采样到 (h',w') 拍平。"""
    full = F.interpolate(p_map, size=size, mode="bilinear", align_corners=False)
    return full.flatten().cpu().numpy()


def make_gallery(root: Path, cls: str, shot: int, seed: int,
                 model: WinCLIP) -> None:
    """从该类 train(全正常)按种子采 shot 张,预计算 gallery 特征。"""
    rng = np.random.default_rng(seed)
    train_imgs = mvtec.iter_train_images(root, cls)
    picks = [train_imgs[i] for i in
             rng.choice(len(train_imgs), size=shot, replace=False)]
    imgs = torch.stack([
        model.preprocess(Image.open(p).convert("RGB")) for p in picks
    ]).to(model.device)
    model.set_gallery(imgs)


@torch.no_grad()
def main():
    args = parse_args()
    root = Path(args.data_root)
    classes = MVTEC_CLASSES if args.classes == "all" else \
        [c.strip() for c in args.classes.split(",")]
    shots_list = [int(s) for s in args.shots.split(",")]

    exp = {
        "script": "evaluate.py", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag, **vars(args), "classes": classes,
    }
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"exp_{time.strftime('%Y%m%d_%H%M%S')}.json"

    model = WinCLIP("ViT-B-16-plus-240", args.weights, args.device)
    print(f"[model] ViT-B-16-plus-240 + weights={args.weights} on {args.device} | "
          f"CLIP 温度 = {model.temp.item():.2f}", flush=True)

    per_shot = {}
    for shot in shots_list:
        per_class = {}
        for cls in classes:
            t0 = time.time()
            model.set_class(cls.replace("_", " "))   # 官方直接用裸类名
            if shot > 0:
                make_gallery(root, cls, shot, args.seed, model)

            scores, labels = [], []
            pix_scores, pix_gts = [], []
            for _, rel_type, img_path, mask_path in \
                    mvtec.iter_test_images(root, cls):
                t = model.preprocess(Image.open(img_path).convert("RGB")) \
                    .unsqueeze(0).to(args.device)
                p_map, img_score = model.anomaly_maps(t, use_few=shot > 0)
                scores.append(img_score)
                labels.append(1 if rel_type != "good" else 0)

                if mask_path is not None:
                    mask = np.asarray(Image.open(mask_path).convert("L")
                                      .resize((240, 240), Image.BILINEAR)) > 128
                    pix_scores.append(up_to_gt(p_map, mask.shape))
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
            print(f"[s{shot} {cls:10s}] image={img_auc*100:6.2f}%  "
                  f"pixel={pix_auc*100:6.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        img_mean = np.nanmean([r["img_auroc"] for r in per_class.values()])
        pix_mean = np.nanmean([r["pix_auroc"] for r in per_class.values()])
        per_shot[str(shot)] = {"per_class": per_class,
                               "mean_img_auroc": round(float(img_mean), 1),
                               "mean_pix_auroc": round(float(pix_mean), 1)}
        print(f"\n==== shot={shot}  {len(classes)} 类均值:"
              f"image {img_mean:.1f}% | pixel {pix_mean:.1f}% ====\n", flush=True)

    exp["results"] = per_shot
    log_path.write_text(json.dumps(exp, ensure_ascii=False, indent=2))
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
