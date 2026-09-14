"""本地导出 OpenVINO IR:原始 CLIP ckpt → .xml/.bin(Intel CPU 部署格式)。

只依赖 CPU torch(无需 GPU)+ open_clip 加载权重,导出后可用 OpenVINO
Runtime 在本机直接推理。与 winclip.py 手工前向一一对应:

  图1 patcher     : (1,3,240,240) 图像 → (1,226,896) [conv1 + [CLS] + 位置 + ln_pre]
  图2 tower_l226  : (1,226,896) → (1,226,640) 整图 12 层(CLS 与 225 patch 特征)
  图3 tower_w5    : (B,5,896)  → (B,5,640)   2×2 窗口子序列,批动态 [1,196]
  图4 tower_w10   : (B,10,896) → (B,10,640)  3×3 窗口子序列,批动态 [1,169]

窗口塔 batch 维不参与任何依赖形状的算子(L 固定、attention softmax 沿 L 维),
导出时把 batch 维放宽为 Dimension(1, hi):同一份 IR 既支持"全批 196/169 窗"
也支持"ROI 子集任意窗口数"(混合流水线的传统 CV 前置只算覆盖窗口)。

数值要点(探针逐层定位):ov.save_model 本版本默认 compress_to_fp16,会把
conv1 等大权重压成 fp16 → 落盘 IR 对拍误差 ~1.6e-3(内存 compile 同图仅
~1e-6,曾误判为 OV 的 LN 降级)。导出统一显式 compress_to_fp16=False,
fp32 全链路下各 IR 对拍 ~1e-6。

导出中间路由:torch.onnx.export → ov.convert_model(ONNX)。不直接用 OV 的
PyTorch 前端(内部 torch.jit.trace sanity check 对含 nn.MultiheadAttention
的塔报 "Graphs differed across invocations");torch.onnx.export 与远程 ONNX
路径同源(实测对拍 ~1e-6)。

窗口索引 / 文本打分等轻逻辑留在外层 python(部署版推理器),模型引擎只管
特征计算——与 winclip.py 结构同构,便于逐 token 对齐验证。

用法:
    python scripts/export_openvino_local.py [--ckpt data/weights/...pt] [--out data/deploy]
输出: <out>/{patcher,tower_l226,tower_w5,tower_w10}.xml/.bin + win_idx_k{2,3}.npy
验证: 每个图导出后立即用本地 CPU torch 对拍,打印最大绝对误差
      (静态全批 + 动态部分批各抽查一次,fp32 链路实测 ~1e-6,tol 1e-3
      只挡结构性错误;端到端判据 = AUROC 漂移 ≤±0.3pt)。
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


def _export_via_onnx(unit: torch.nn.Module, example: torch.Tensor,
                     onnx_tmp: Path, batch_dyn: bool) -> ov.Model:
    """导出中间路由:torch.onnx.export → ov.convert_model(ONNX)。

    不直接用 ov.convert_model(module):其内部 torch.jit.trace 的 sanity check
    对含 nn.MultiheadAttention 的塔会报 "Graphs differed across invocations"
    (MHA 数据相关的 kernel 路径选择),且 ln_pre 对拍误差 ~1e-3 偏大;
    torch.onnx.export 无该检查、远程 ONNX 路径实测对拍 ~1e-6,同源更稳。

    batch_dyn=True 时第 0 维放宽为 dynamic(窗口塔),L 与 896 恒静态。
    """
    # dynamo=False:legacy TorchScript 导出(与远程 ONNX 路径同源,不依赖 onnxscript)。
    # 注:torch 2.14 的 legacy 导出器 dynamic_axes 需按输入名 key(按 0 索引报错)
    dyn = {"x": {0: "batch"}} if batch_dyn else None
    torch.onnx.export(unit, example, onnx_tmp, opset_version=17,
                      input_names=["x"], output_names=["y"],
                      dynamic_axes=dyn, dynamo=False)
    m = ov.convert_model(onnx_tmp)
    if batch_dyn:
        _relax_batch(m, example.shape[0], example.shape[1])
    return m


def _relax_batch(model: ov.Model, batch_hi: int, L: int) -> None:
    """窗口塔:batch 维放宽为 Dimension(1, batch_hi),L 与 896 保持静态。

    reshape 后保存,IR 内即动态 batch;同一份文件同时服务全批与 ROI 子集。
    batch_hi/L 由导出形状直接传入(get_partial_shape 返回 Dimension 对象,
    无 get_length 时不能 int() 强转,版本间 API 不稳,不解析)。
    """
    iname = model.input(0).get_any_name()
    ps = ov.PartialShape([ov.Dimension(1, batch_hi), L, 896])
    model.reshape({iname: ps})


def _dump_window_indices(out: Path) -> None:
    """15×15 网格滑窗 token 索引,与 winclip._window_indices 逐位一致。

    产物供部署运行时使用(零 torch 依赖);(196,4)/(169,9),token 号 1..225。
    """
    for k, fname in ((2, "win_idx_k2.npy"), (3, "win_idx_k3.npy")):
        board = torch.arange(1, 226, dtype=torch.float32).view(1, 1, 15, 15)
        masks = F.unfold(board, kernel_size=k, stride=1).squeeze(0)  # (k², n_win)
        idx = masks.t().long().numpy()
        assert idx.shape == ((15 - k + 1) ** 2, k * k), idx.shape
        np.save(out / fname, idx)
        print(f"[ok] {fname} | 形状 {idx.shape}(token 1..225)", flush=True)


def _check_ov(compiled, example: torch.Tensor, unit: torch.nn.Module,
              label: str, tol: float = 1e-3) -> None:
    """对拍:同一输入 CPU torch 前向 vs OV 输出,打印 max_err。

    tol 说明:fp32 全链路(compress_to_fp16=False)下各 IR 与 torch 同机对拍
    实测 ~1e-6(探针验证:误差主源曾是该版本 save_model 默认 fp16 压缩 conv
    权重,显式关闭后消除)。tol=1e-3 只挡结构性错误(数量级异常);最终判据 =
    端到端 AUROC 漂移 ≤±0.3pt(eval_ov vs evaluate 同机对拍)。
    """
    with torch.no_grad():
        ref = unit(example)
    got = torch.from_numpy(compiled(example.numpy())[0])
    err = (got - ref).abs().max().item()
    status = "OK" if err < tol else "FAIL"
    print(f"[{status}] {label:22s} | 输出 {tuple(got.shape)}  "
          f"max_err={err:.2e} (tol {tol:.0e})", flush=True)


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
    torch.manual_seed(0)   # 对拍输入固定,便于复现

    # ---- 图1 patcher:图像 → ln_pre 后 tokens(整图 L=226,静态) ----------
    # compress_to_fp16=False:该版本 save_model 默认把权重压成 fp16,conv1 大
    # 权重相对误差 ~5e-4 → 输出绝对误差 ~1e-3(对拍实测 1.59e-3,塔内每层 LN
    # 会把误差压小所以 tower 只 ~1.6e-4)。显式关掉,保 fp32 逐位对齐(~1e-6)。
    patcher = Patcher(v).eval()
    img = torch.randn(1, 3, 240, 240)
    ov_model = _export_via_onnx(patcher, img, out / "_tmp_patcher.onnx", False)
    ov.save_model(ov_model, out / "patcher.xml", compress_to_fp16=False)
    _check_ov(core.compile_model(out / "patcher.xml", "CPU"), img, patcher, "patcher.xml")

    # ---- 图2 tower:整图序列 L=226(静态) ---------------------------------
    tower = TokTower(v).eval()
    toks_full = torch.randn(1, 226, 896)
    ov_model = _export_via_onnx(tower, toks_full, out / "_tmp_tower_l226.onnx", False)
    ov.save_model(ov_model, out / "tower_l226.xml", compress_to_fp16=False)
    _check_ov(core.compile_model(out / "tower_l226.xml", "CPU"),
              toks_full, tower, "tower_l226.xml")

    # ---- 图3/4 tower:窗口子序列(2×2 → L=5 批196 / 3×3 → L=10 批169) -----
    # batch 维动态:同一 IR 服务全批与 ROI 子集(混合流水线只算覆盖窗口)
    for L, bsize, fname in ((5, 196, "tower_w5.xml"), (10, 169, "tower_w10.xml")):
        full_batch = torch.randn(bsize, L, 896)
        ov_model = _export_via_onnx(tower, full_batch, out / f"_tmp_{fname}.onnx", True)
        ov.save_model(ov_model, out / fname, compress_to_fp16=False)
        compiled = core.compile_model(out / fname, "CPU")
        _check_ov(compiled, full_batch, tower, f"{fname}(全批 {bsize}×{L})")
        # 动态批抽查:非全批形状,证明 ROI 子集任意窗口数可用
        sub_batch = torch.randn(37 if L == 5 else 88, L, 896)
        _check_ov(compiled, sub_batch, tower, f"{fname}(部分批 {sub_batch.shape[0]}×{L})")

    # 清理中间 ONNX(仅用于导 IR,不保留)
    for f in out.glob("_tmp_*.onnx"):
        f.unlink()

    # ---- 部署常量:窗口索引(运行时零 torch) --------------------------------
    _dump_window_indices(out)

    for f in sorted(out.glob("*.xml")):
        b = out / (f.stem + ".bin")
        print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB + "
              f"{b.stat().st_size/1e6:.1f}MB)")
    print(f"[done] 全部导出完成,共 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
