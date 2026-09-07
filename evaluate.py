"""WinCLIP 评估:image / pixel AUROC(逐类 + 15 类均值)。

用法示例:
    python evaluate.py --data_root /root/autodl-tmp/mvtec_anomaly_detection
    python evaluate.py --data_root ... --shots 2 --classes bottle,hazelnut
    python evaluate.py --data_root ... --shots 0,2,4 --mode softmax --tag 消融

实验参数与指标全量写入 logs/exp_<时间戳>.json,参数不进文件名。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

import mvtec
from mvtec import class_prompt_noun, is_texture
from winclip import WinCLIP


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="mvtec_anomaly_detection 目录")
    ap.add_argument("--classes", default="all",
                    help="逗号分隔类别列表;默认 all(15 类)")
    ap.add_argument("--shots", default="0",
                    help="逗号分隔的 few-shot 正常样本数;0=zero-shot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--mode", default="diff", choices=["diff", "softmax"])
    ap.add_argument("--vote_win", type=int, default=3)
    ap.add_argument("--topk_pct", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="实验备注,写入 log")
    return ap.parse_args()


def geometric_transform(img: Image.Image, size: int) -> Image.Image:
    """与图像预处理相同的几何缩放(短边 resize + center crop)。"""
    img = img.convert("RGB") if img.mode != "L" else img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)  # MVTec 多为方形,直接拉伸可对齐 mask
    return img


@torch.no_grad()
def main():
    args = parse_args()
    root = Path(args.data_root)
    classes = mvtec.MVTEC_CLASSES if args.classes == "all" else \
        [c.strip() for c in args.classes.split(",")]
    shots_list = [int(s) for s in args.shots.split(",")]

    exp = {
        "script": "evaluate.py", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag, **vars(args), "classes": classes,
    }
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"exp_{time.strftime('%Y%m%d_%H%M%S')}.json"

    model = WinCLIP(args.model_name, args.pretrained, args.device,
                    args.image_size, args.mode, args.vote_win, args.topk_pct)
    print(f"[model] {args.model_name}/{args.pretrained} on {args.device} | "
          f"mode={args.mode} vote_win={args.vote_win} topk={args.topk_pct}")

    overall = {}
    for shot in shots_list:
        results = {}
        for cls in classes:
            noun = class_prompt_noun(cls)
            model.set_class(noun)
            if shot > 0:  # few-shot:从 train(全正常)固定 seed 采 k 张
                rng = np.random.default_rng(args.seed)
                train_imgs = mvtec.iter_train_images(root, cls)
                picks = [train_imgs[i] for i in
                         rng.choice(len(train_imgs), size=shot, replace=False)]
                refs = []
                for p in picks:
                    t = model.preprocess(geometric_transform(Image.open(p).convert("RGB"),
                                                             args.image_size)).unsqueeze(0).to(args.device)
                    refs.append(model.encode_patches(t))
                model.ref_feats = torch.cat(refs, dim=0)  # (k*49, D)
            else:
                model.ref_feats = None

            scores, labels = [], []
            pix_scores, pix_gts = [], []
            n_good = n_defect = 0
            for name, rel_type, img_path, mask_path in mvtec.iter_test_images(root, cls):
                img = Image.open(img_path).convert("RGB")
                t = model.preprocess(geometric_transform(img, args.image_size)) \
                    .unsqueeze(0).to(args.device)
                p_map, img_score = model.anomaly_maps(t)
                scores.append(img_score)
                labels.append(1 if rel_type != "good" else 0)
                n_good += rel_type == "good"
                n_defect += rel_type != "good"

                if mask_path is not None:  # pixel 级:缺陷图 + GT mask
                    mask = geometric_transform(Image.open(mask_path), args.image_size)
                    mask = np.asarray(mask) > 128
                    side = int(p_map.numel() ** 0.5)
                    grid = p_map.view(1, 1, side, side)
                    full = torch.nn.functional.interpolate(
                        grid, size=(mask.shape[0], mask.shape[1]),
                        mode="bilinear", align_corners=False)
                    pix_scores.append(full.flatten().cpu().numpy())
                    pix_gts.append(mask.flatten())

            img_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
            pix_auc = (roc_auc_score(np.concatenate(pix_gts), np.concatenate(pix_scores))
                       if pix_gts else float("nan"))
            results[cls] = {"n_good": n_good, "n_defect": n_defect,
                            "img_auroc": round(img_auc * 100, 1),
                            "pix_auroc": round(pix_auc * 100, 1)}
            print(f"[shot={shot}] {cls:10s} img={img_auc*100:5.1f}%  "
                  f"pix={pix_auc*100:5.1f}%  (good {n_good} / defect {n_defect})")

        img_mean = np.mean([r["img_auroc"] for r in results.values()])
        pix_mean = np.mean([r["pix_auroc"] for r in results.values()])
        overall[str(shot)] = {"per_class": results,
                              "mean_img_auroc": round(float(img_mean), 1),
                              "mean_pix_auroc": round(float(pix_mean), 1)}
        print(f"\n==== shot={shot}  15 类均值:image AUROC {img_mean:.1f}% | "
              f"pixel AUROC {pix_mean:.1f}% ====\n")

    exp["results"] = overall
    log_path.write_text(json.dumps(exp, ensure_ascii=False, indent=2))
    print(f"[log] 参数与指标已写入 {log_path}")


if __name__ == "__main__":
    main()
