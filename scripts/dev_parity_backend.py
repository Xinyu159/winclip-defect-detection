"""后端对拍门:torch WinCLIP(CPU 参考) vs 部署后端(OV/ORT)逐值比对。

与 dev_parity.py 的关系:
    dev_parity.py 把 OVEngine 写死在 import 里,只能在装了 OpenVINO 的机器跑。
    本脚本按 --deploy 目录自动选后端,判据、容差、检查项**完全照抄**它 ——
    多一条后端不该放松验收标准,否则"换了后端"就成了精度悄悄下降的借口。

覆盖四个层面,任一 FAIL 即 exit 1:
  1. preprocess:真实图,torch transform vs numpy 镜像的 (1,3,240,240)。
  2. zero-shot anomaly_maps:map 逐元素 max|Δ|、img_score |Δ|。
  3. few-shot:参考正常图 gallery + 查询图,map / img_score 同判据。
  4. 窗口塔动态 batch:同一图,全量一次 / 分块多次喂 —— 结果必须一致。
     (这条是 ONNX 动态 batch 的专属风险:静态图在标称形状也跑得通,
      "能跑"不等于"动态维真的有效"。)

判据:map <1e-3、img_score <1e-4、preprocess <5e-5(与 dev_parity.py 同)。
权威判据仍是端到端 AUROC 漂移 ≤±0.3pt(eval_ov 各档位对比)。

用法:
    python scripts/dev_parity_backend.py --deploy data/deploy_onnx_dyn \
        --device cuda --classes carpet,tile --n-random 3
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
from runtime.pipeline import OVPipeline                   # noqa: E402
from scripts.eval_ov import build_engine                  # noqa: E402


def img_files(root: Path, n: int, seed: int = 0) -> list[Path]:
    """从 MVTec bottle 的 test 里抽 n 张(存在性好,内容随机)。"""
    rng = np.random.default_rng(seed)
    imgs = sorted((root / "bottle" / "test").rglob("*.png"))
    picks = [imgs[i] for i in rng.choice(len(imgs), size=n, replace=False)]
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="carpet,tile")
    ap.add_argument("--deploy", default="data/deploy_onnx_dyn")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--img-src", default="data/mvtec_anomaly_detection")
    ap.add_argument("--ckpt",
                    default="data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    ap.add_argument("--tf32", action="store_true",
                    help="对拍时开启 TF32(默认关:对拍要测实现正确性,"
                         "不该被 GPU 的 ~1e-3 舍入噪声掩盖)")
    args = ap.parse_args()

    fails: list[str] = []

    def check(name: str, v: float, tol: float, note: str = "") -> None:
        ok = v < tol
        print(f"[{'OK' if ok else 'FAIL'}] {name:34s} max|Δ|={v:.2e} "
              f"(tol {tol:.0e}){('  ' + note) if note else ''}", flush=True)
        if not ok:
            fails.append(name)

    torch.manual_seed(0)
    wc = WinCLIP(weights=args.ckpt, device="cpu")            # torch 参考
    engine, backend, device = build_engine(args.deploy, args.device,
                                           tf32=args.tf32)
    ovp = OVPipeline(engine, args.text)
    src = Path(args.img_src)
    imgs = [Image.open(p).convert("RGB") for p in img_files(src, args.n_random)]
    t0 = time.time()
    print(f"[engine] {backend} | device={device} | "
          f"tf32={'on' if args.tf32 else 'off'}", flush=True)

    # 预处理参考口径。WinCLIP.preprocess = open_clip 的 **preprocess_train**
    # (RandomResizedCrop,每次调用随机),不是评估用的 val transform。部署侧
    # EngineBase.preprocess_rgb 是**确定性** resize(方形输入即 Resize240+
    # CenterCrop,与 GT mask 的 resize 同几何)。所以对拍必须拿 val 作参考;
    # 用 train 作参考会永远失败,那个"失败"反映的是 research 路径的口径问题,
    # 不是部署实现的错。这里两条都测:val = 判定项,train = 信息项。
    import open_clip
    _, pre_train, pre_val = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained=args.ckpt, device="cpu")

    for cls in [c.strip() for c in args.classes.split(",")]:
        cname = cls.replace("_", " ")
        wc.set_class(cname)
        ovp.set_class(cname)

        # ---- 1. preprocess ------------------------------------------
        errs_val = [float(np.abs(pre_val(p).numpy()
                                 - engine.preprocess_rgb(np.asarray(p))[0]).max())
                    for p in imgs]
        check(f"[{cls}] preprocess vs val {len(errs_val)} 图", max(errs_val), 5e-5)
        errs_tr = [float(np.abs(pre_train(p).numpy()
                                - engine.preprocess_rgb(np.asarray(p))[0]).max())
                   for p in imgs]
        print(f"  [note] vs train(随机裁剪,research 现状) max|Δ|="
              f"{max(errs_tr):.2e} —— 与 val 不同源,属口径差异不是实现误差",
              flush=True)

        # ---- 2. zero-shot -------------------------------------------
        # 两条前向喂**同一个张量**(部署侧的确定性预处理输出),这样比的纯粹是
        # "torch 塔 vs ORT 塔"的数值一致性,不掺预处理口径的差异。
        for i, pil in enumerate(imgs):
            x = torch.from_numpy(engine.preprocess_rgb(np.asarray(pil)))
            m_t, s_t = wc.anomaly_maps(x)                    # (1,1,15,15), float
            m_o, s_o, _ = ovp.anomaly_maps(x.numpy())
            check(f"[{cls}] zero map 图{i}", float(np.abs(
                m_t[0, 0].numpy() - m_o).max()), 1e-3)
            check(f"[{cls}] zero score 图{i}", abs(s_t - s_o), 1e-4)

        # ---- 3. few-shot --------------------------------------------
        train_imgs = sorted((src / cls / "train" / "good").glob("*.png"))
        if train_imgs:
            gal = np.concatenate([engine.preprocess_rgb(
                np.asarray(Image.open(p).convert("RGB")))
                for p in train_imgs[:2]], axis=0)
            wc.set_gallery(torch.from_numpy(gal))
            ovp.set_gallery(gal)
            x = torch.from_numpy(engine.preprocess_rgb(
                np.asarray(imgs[0])))
            m_t, s_t = wc.anomaly_maps(x, use_few=True)
            m_o, s_o, _ = ovp.anomaly_maps(x.numpy(), use_few=True)
            check(f"[{cls}] few map 图0", float(np.abs(
                m_t[0, 0].numpy() - m_o).max()), 1e-3)
            check(f"[{cls}] few score 图0", abs(s_t - s_o), 1e-4)

        # ---- 4. 窗口塔动态 batch(后端专属风险)---------------------
        # 同一张图:整块喂 196/169 个窗口,vs 分块喂(B=7)。若动态维是假的,
        # 分块那次会算错(补零/复用标称形状),而"只喂标称形状"的常规测试
        # 根本发现不了。这里直接比块间一致性。
        x = engine.preprocess_rgb(np.asarray(imgs[0]))
        toks = engine.patcher(x)
        for k, n_win in ((2, 196), (3, 169)):
            L = k * k + 1
            idx = engine.window_indices(k)
            seq = np.concatenate(
                [np.zeros((n_win, 1), dtype=np.int64), idx], axis=1)
            rows = toks[0][seq].astype(np.float32)           # (n_win,L,896)
            whole = engine.tower_w(rows)                     # 一次全量
            parts = np.concatenate(
                [engine.tower_w(rows[s:s + 7]) for s in range(0, n_win, 7)],
                axis=0)                                      # 分块累积
            d = float(np.abs(whole - parts).max())
            check(f"[{cls}] k={k} 动态batch 分块一致({n_win}→7/块)", d, 1e-5)

    print(f"\n[parity] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}"
          f" | {time.time()-t0:.0f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
