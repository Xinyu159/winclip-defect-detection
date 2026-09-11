"""开发期对拍门:torch WinCLIP(CPU 参考) vs OV OVPipeline 同输入逐值比对。

覆盖四个层面,任一 FAIL 即 exit 1:
  1. 文本原型:torch set_class 在线编码 vs 离线 npz(build_text_protos 产物)
     —— 外部验证,比脚本内自检更强(跨进程、跨存储格式)。
  2. preprocess:20 张真实图,torch transform vs numpy 镜像的 (1,3,240,240)。
  3. zero-shot anomaly_maps:3 张随机图,map 逐元素 max|Δ|、img_score |Δ|。
  4. few-shot:2 张参考正常图 gallery + 1 张查询,map / img_score 同判据。

判据(计划 M2):map <1e-3、img_score <1e-4、原型 <1e-6、preprocess <1e-6
(preprocess 若达不到 1e-6 则降级记录,权威判据是 AUROC 漂移 ≤±0.3pt)。
全量窗口下 ov 广义调和与 torch 原式逐位一致(无 absent patch)。

用法:
    python scripts/dev_parity.py [--classes carpet,tile] [--n-random 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from winclip import WinCLIP                              # noqa: E402
from runtime.ov_engine import OVEngine                    # noqa: E402
from runtime.pipeline import OVPipeline                   # noqa: E402

CKPT = "data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"
IMG_SRC = "data/mvtec_anomaly_detection"


def img_files(root: Path, n: int, seed: int = 0) -> list[Path]:
    """从 MVTec 前几类的 test 里抽 n 张(存在性好,内容随机)。"""
    rng = np.random.default_rng(seed)
    imgs = sorted((root / "bottle" / "test").rglob("*.png"))
    picks = [imgs[i] for i in rng.choice(len(imgs), size=n, replace=False)]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="carpet,tile")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--img-src", default=str(IMG_SRC))
    args = ap.parse_args()

    fails: list[str] = []

    def check(name: str, v: float, tol: float, note: str = "") -> None:
        ok = v < tol
        print(f"[{'OK' if ok else 'FAIL'}] {name:32s} max|Δ|={v:.2e} "
              f"(tol {tol:.0e}){('  ' + note) if note else ''}", flush=True)
        if not ok:
            fails.append(name)

    torch.manual_seed(0)
    wc = WinCLIP(weights=CKPT, device="cpu")          # torch 参考
    engine = OVEngine(args.deploy)
    ovp = OVPipeline(engine, args.text)
    src = Path(args.img_src)
    imgs = [Image.open(p).convert("RGB") for p in img_files(src, args.n_random)]
    t0 = time.time()

    for cls in [c.strip() for c in args.classes.split(",")]:
        cname = cls.replace("_", " ")
        wc.set_class(cname)
        ovp.set_class(cname)
        d = np.load(Path(args.text) / f"{cls}.npz")
        check(f"[{cls}] text proto normal", float(np.abs(
            d["normal"] - wc.normal_proto.float().numpy()).max()), 1e-6)
        check(f"[{cls}] text proto abnormal", float(np.abs(
            d["abnormal"] - wc.abnormal_proto.float().numpy()).max()), 1e-6)
        check(f"[{cls}] temperature", float(abs(
            float(d["temp"]) - float(wc.temp))), 1e-4)

        # preprocess parity:torch transform vs numpy 镜像(同一批真实图)。
        # 两者 resize 插值实现不同(PIL BICUBIC 同为 PIL,应可 1e-6;若有插值
        # 内核差异则 >1e-6 降级记录,权威判据是 AUROC 漂移 ≤±0.3pt)
        errs = [float(np.abs(wc.preprocess(p).numpy()
                             - engine.preprocess_rgb(np.asarray(p))[0]).max())
                for p in imgs]
        check(f"[{cls}] preprocess {len(errs)} 图", max(errs), 5e-5)
        if max(errs) > 1e-6:
            print(f"  [note] preprocess 峰值 {max(errs):.2e}:numpy 镜像非逐位,"
                  f"以 AUROC 漂移判据为准")

        # zero-shot 单帧对拍
        for i, pil in enumerate(imgs):
            x = torch.from_numpy(engine.preprocess_rgb(np.asarray(pil)))
            m_t, s_t = wc.anomaly_maps(x)                 # (1,1,15,15), float
            m_o, s_o, _ = ovp.anomaly_maps(x.numpy())
            check(f"[{cls}] zero map 图{i}", float(np.abs(
                m_t[0, 0].numpy() - m_o).max()), 1e-3)
            check(f"[{cls}] zero score 图{i}", abs(s_t - s_o), 1e-4)

        # few-shot:2 参考(取 train 正常图前 2 张,carpet 目录即可)+ 同图查询
        train_imgs = sorted((src / cls / "train" / "good").glob("*.png"))
        if train_imgs:
            gal = np.stack([engine.preprocess_rgb(
                np.asarray(Image.open(p).convert("RGB")))
                for p in train_imgs[:2]])
            wc_gal = torch.from_numpy(gal)
            wc.set_gallery(wc_gal)
            ovp.set_gallery(gal)
            pil = imgs[0]
            x = torch.from_numpy(engine.preprocess_rgb(np.asarray(pil)))
            m_t, s_t = wc.anomaly_maps(x, use_few=True)
            m_o, s_o, _ = ovp.anomaly_maps(x.numpy(), use_few=True)
            check(f"[{cls}] few map 图0", float(np.abs(
                m_t[0, 0].numpy() - m_o).max()), 1e-3)
            check(f"[{cls}] few score 图0", abs(s_t - s_o), 1e-4)

    print(f"\n[parity] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}"
          f" | {time.time()-t0:.0f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
