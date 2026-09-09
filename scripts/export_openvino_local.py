"""本地导出 OpenVINO IR:原始 CLIP ckpt → .xml/.bin(Intel CPU 部署格式)。

只依赖 CPU torch(无需 GPU)+ open_clip 加载权重,导出后可用 OpenVINO
Runtime 在本机直接推理。与 winclip.py 手工前向一一对应:

  图1 patcher : (1,3,240,240) 图像 → (1,226,896)  [conv1 + [CLS] + 位置 + ln_pre]
  图2 tower   : (B,L,896) token 序列 → (B,L,640)  [12 blocks + ln_post + proj + l2]
                  L 动态:整图 226 或窗口子序列 k²+1

窗口索引 / 文本打分等轻逻辑留在外层 python(部署版推理器),模型引擎只管
特征计算——与 winclip.py 结构同构,便于逐 token 对齐验证。

用法:
    python export_openvino_local.py [--ckpt data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt]
                                    [--out data/deploy]
输出: <out>/patcher.xml/.bin 与 <out>/tower.xml/.bin(OpenVINO IR)
验证: 每个图导出后立即用本地 CPU torch 对拍,打印最大绝对误差。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import open_clip
import openvino as ov


# ---- 导出单元:与 winclip.py 手工前向相同的模块组合 ----------------------
class Patcher(torch.nn.Module):
    """(1,3,240,240) → (1,226,896):conv1 patch 化 + [CLS] + 位置编码 + ln_pre。"""

    def __init__(self, v):
        super().__init__()
        self.conv1 = v.conv1
        self.class_embedding = v.class_embedding
        self.positional_embedding = v.positional_embedding
        self.ln_pre = v.ln_pre

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # conv1:(B,896,15,15) → 按 patch_dim 整形为 (B,225,896),即 (B, 通道, 空间) 转置
        x = self.conv1(img).reshape(img.shape[0], 896, -1).permute(0, 2, 1)
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding
        return self.ln_pre(x)


class TokTower(torch.nn.Module):
    """(B,L,896) → (B,L,640):12 blocks + ln_post + proj,逐 token l2。
    L 任意(整图 226 / 窗口 2×2 / 3×3),窗口级只用 CLS token。"""

    def __init__(self, v):
        super().__init__()
        self.blocks = v.transformer.resblocks
        self.ln_post = v.ln_post
        self.proj = v.proj

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = tokens
        for block in self.blocks:      # resblocks 为 ModuleList,逐块调用以支持导出
            h = block(h)
        h = self.ln_post(h)
        return F.normalize(h @ self.proj, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    ap.add_argument("--out", default="data/deploy")
    args = ap.parse_args()

    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained=args.ckpt, device="cpu")
    v = model.visual
    print(f"[load] ckpt 加载完成 {time.time()-t0:.1f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    core = ov.Core()

    # ---- 图1 patcher:图像 → ln_pre 后 tokens -----------------------------
    patcher = Patcher(v).eval()
    img = torch.randn(1, 3, 240, 240)
    ov_model = ov.convert_model(patcher, example_input=img)
    ov.save_model(ov_model, out / "patcher.xml")
    # torch 对拍
    with torch.no_grad():
        ref = patcher(img)
    compiled = core.compile_model(out / "patcher.xml", "CPU")
    got = torch.from_numpy(compiled(img.numpy())[0])
    print(f"[ok] patcher.xml  | 输出 {tuple(got.shape)}  "
          f"max_err={ (got - ref).abs().max().item():.2e}", flush=True)

    # ---- 图2 tower:任意长度 token 序列 → 640 共享空间 --------------------
    tower = TokTower(v).eval()
    toks = torch.randn(1, 226, 896)
    ov_model = ov.convert_model(tower, example_input=toks)
    ov.save_model(ov_model, out / "tower.xml")
    with torch.no_grad():
        ref = tower(toks)
    compiled = core.compile_model(out / "tower.xml", "CPU")
    got = torch.from_numpy(compiled(toks.numpy())[0])
    print(f"[ok] tower.xml    | 输出 {tuple(got.shape)}  "
          f"max_err={ (got - ref).abs().max().item():.2e}", flush=True)

    # ---- 动态长度抽查:窗口子序列(2×2 → L=5)也走同一个 tower ---------------
    win = torch.randn(169, 5, 896)
    ov_model = ov.convert_model(tower, example_input=win)
    ov.save_model(ov_model, out / "tower_win.xml")
    compiled = core.compile_model(out / "tower_win.xml", "CPU")
    with torch.no_grad():
        ref = tower(win)
    got = torch.from_numpy(compiled(win.numpy())[0])
    print(f"[ok] tower_win.xml| 窗口批 169×5 输出 {tuple(got.shape)}  "
          f"max_err={ (got - ref).abs().max().item():.2e}", flush=True)

    for f in sorted(out.glob("*.xml")):
        print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB + "
              f"{(out/f.name.replace('.xml','.bin')).stat().st_size/1e6:.1f}MB)")
    print(f"[done] 全部导出完成,共 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
