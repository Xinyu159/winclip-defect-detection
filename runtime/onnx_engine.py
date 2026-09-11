"""ONNX Runtime 引擎:ONNX 加载 + EP 选择,接口与 OVEngine 逐一对齐。

与 ov_engine.py 的关系:
    两者都继承 EngineBase(preprocess + 窗口索引**同源同算法**),差别只在
    "推理怎么跑":OVEngine 走 OpenVINO 编译 IR(Intel 平台),本引擎走
    onnxruntime 的 ExecutionProvider(NVIDIA/TensorRT 平台)。**打分数学
    完全不在引擎里**——引擎只管把张量喂进去、把结果取出来,所以两条链路
    在 pipeline.OVPipeline 里共享同一套调和/softmax/最近邻。

I/O 约定照抄远程已验证的 export_onnx.py(对拍 max_err ~1e-6):
    patcher.onnx        (1,3,240,240)  → (1,226,896)
    tower.onnx          (1,226,896)    → (1,226,640)
    tower_win2x2.onnx   (B,5,896)      → (B,5,640)   batch 动态
    tower_win3x3.onnx   (B,10,896)     → (B,10,640)  batch 动态

窗口塔的 batch 动态是本引擎的存在理由:级联按预算只算 top-N 窗口,
静态 batch 必须补零到 196/169,白烧算力 —— 正是级联想省掉的那部分。

EP 选择:
    "auto"(默认) CUDA → CPU 依次回落,并打印实际生效的 EP(不静默降级)
    "cuda:0" / "cpu" 等 显式指定,不可用时直接报错

TF32(NVIDIA 20 系及以上):
    CUDA EP 默认开启 TF32(10 位尾数 ≈1e-3 相对误差)。实测同一模型不同 batch
    大小会选中不同 kernel,舍入顺序不同 → 分块推理与整批推理出现 3e-4~3e-3 的
    差异(scripts/dev_dynbatch_probe.py 实测:CPU 精确 0,TF32 开 3.2e-4,
    TF32 关 6.6e-7)。这不是动态维错了,是 GPU 数值精度问题。**需要逐位对齐
    的场合(对拍/漂移归因)必须 tf32=False**;追求吞吐时用默认(开启)。

注意 CUDA EP 在本环境需要 LD_LIBRARY_PATH 指向 nvidia/*/lib(见 README
踩坑速查);否则会报 libcudnn.so.9 缺失。构造时会给出可执行的修复指引。
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .engine_base import EngineBase

_LD_PREPENDED = False


def _prepend_nvidia_libs() -> None:
    """把 pip 装的 nvidia/*/lib 挂进加载器搜索路径,让 CUDA EP 找得到 cudnn。

    背景:onnxruntime-gpu 要求 libcudnn.so.9 / libcublas.so.12 在进程的库搜索
    路径上,但 `pip install nvidia-cudnn-cu12` 只把 .so 放进 site-packages,
    **不改 LD_LIBRARY_PATH**。结果是 import 一切正常、CUDA EP 静默回落到 CPU
    (或报 "libcudnn.so.9: cannot open shared object file")——本仓库实测踩到
    过,task C 的 GPU 节拍差点被测成 CPU 数字。

    两种生效手段,这里都用上:
      - os.environ:子进程(以及 dlopen 内部再 dlopen)能继承
      - ctypes.CDLL 预加载:本进程 dlopen 时若无 rpath,靠 RTLD_GLOBAL 兜住
        依赖链(cudnn → cublas/cudart),顺序错会报 "undefined symbol"

    只对已存在且尚未在路径上的目录动手;找不到任何 nvidia 目录就静默返回,
    CPU-only 安装不该因为这个报错。
    """
    global _LD_PREPENDED
    if _LD_PREPENDED:
        return
    _LD_PREPENDED = True

    # 依赖顺序:先底层运行库,再 cudnn(它依赖 cublas/cudart)
    order = ("cuda_runtime", "cublas", "cufft", "cudnn")
    base = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "nvidia"
    dirs: list[str] = []
    for name in order:
        d = base / name / "lib"
        if d.is_dir():
            dirs.append(str(d))
    # 非标准布局(torch 自带/系统 nvidia 包)兜底:按已知 soname 找
    if not dirs:
        dirs = sorted({str(Path(p).parent) for p in
                       glob.glob(os.path.join(sys.prefix, "**", "libcudnn.so.*"),
                                 recursive=True)})
    if not dirs:
        return

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    new = [d for d in dirs if d not in parts]
    if new:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new + parts)
    for d in dirs:
        for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass        # 非库文件/架构不符:交给 ORT 自己报错,信息更准

#: 逻辑名 → 文件名(与导出脚本一致)
ONNX_FILES = {"patcher": "patcher.onnx", "full": "tower.onnx",
              "w5": "tower_win2x2.onnx", "w10": "tower_win3x3.onnx"}

#: 窗口序列长度 → 逻辑名
_WIN_KEY = {5: "w5", 10: "w10"}

# 导入即挂库路径:ORT 在 import 时就做一次 EP 探测,晚于这时候再挂,
# get_available_providers() 可能已经把 CUDA EP 记为不可用。
_prepend_nvidia_libs()


class OnnxEngine(EngineBase):
    """加载 deploy_dir 下的 4 个 ONNX,惰性建会话,提供与 OVEngine 同名的接口。"""

    def __init__(self, deploy_dir: str | Path, device: str = "auto",
                 providers: list | None = None, tf32: bool | None = None,
                 sess_options: ort.SessionOptions | None = None):
        super().__init__(deploy_dir)
        self.device = device
        self.tf32 = tf32
        self._so = sess_options
        self._providers = providers or self._resolve_providers(device, tf32)
        self._sess: dict[str, ort.InferenceSession] = {}
        self._inames: dict[str, str] = {}

    # ------------------------------------------------------------------
    # EP 解析
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_providers(device: str, tf32: bool | None = None) -> list:
        """把 device 意图解析成 ORT provider 列表(带 CPU 兜底,便于诊断)。

        tf32 非 None 时给 CUDA EP 挂 provider option —— 需用元组形式传入,
        ORT 才认("use_tf32" 是 CUDA EP 的私有选项,不是全局 SessionOptions)。
        """
        # 必须在 get_available_providers() 之前:该调用本身就会尝试 dlopen
        # CUDA EP,库找不到的话列表里根本不会出现它,连报错都看不到。
        if device == "auto" or device.startswith("cuda") or device == "gpu" \
                or "Tensorrt" in device:
            _prepend_nvidia_libs()
        avail = ort.get_available_providers()
        if device == "auto" or device.startswith("cuda") or device == "gpu":
            order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            order = ["CPUExecutionProvider"]
        else:
            # 允许直接传 provider 名(如 TensorrtExecutionProvider)
            order = [device, "CPUExecutionProvider"]
        picked = [p for p in order if p in avail]
        if not picked:
            raise RuntimeError(
                f"无可用的 ORT provider: 请求 {device},"
                f"当前可用 {avail}")
        if tf32 is not None:
            picked = [
                (p, {"use_tf32": "1" if tf32 else "0"})
                if p == "CUDAExecutionProvider" else p
                for p in picked]
        return picked

    def active_provider(self) -> str:
        """实际生效的 EP(取第一个会话的)。不会静默回落到 CPU。"""
        for key in ("patcher", "full", "w5", "w10"):
            if key in self._sess:
                return self._sess[key].get_providers()[0]
        return self._sess.setdefault(
            "patcher", self._create("patcher")).get_providers()[0]

    # ------------------------------------------------------------------
    def _create(self, key: str) -> ort.InferenceSession:
        path = self.dir / ONNX_FILES[key]
        if not path.exists():
            avail = sorted(p.name for p in self.dir.glob("*.onnx"))
            raise FileNotFoundError(
                f"ONNX 模型缺失: {path}\n"
                f"  目录内现有: {', '.join(avail) if avail else '(无 .onnx)'}\n"
                f"  → 导出: python scripts/export_onnx_dyn.py")
        try:
            sess = ort.InferenceSession(str(path), sess_options=self._so,
                                        providers=self._providers)
        except Exception as e:                              # noqa: BLE001
            msg = str(e)
            if "cudnn" in msg.lower() or "cublas" in msg.lower():
                raise RuntimeError(
                    f"CUDA EP 动态库缺失: {msg}\n"
                    f"  → export LD_LIBRARY_PATH=<env>/lib/python3.10/"
                    f"site-packages/nvidia/{{cudnn,cublas,cuda_runtime}}/lib"
                ) from e
            raise
        # EP 回落的可见性:请求了 CUDA 却拿到 CPU 属于**静默降级**,
        # 会把"GPU 节拍"测成 CPU 数字 —— 这是最危险的一类测量错误。
        got = sess.get_providers()[0]
        want = self._providers[0]
        want = want[0] if isinstance(want, tuple) else want
        if want == "CUDAExecutionProvider" and got != "CUDAExecutionProvider":
            print(f"[warn] 请求 CUDA 但实际生效 {got}"
                  f"(节拍数字不可当 GPU 用!)", flush=True)
        # 输入名在 _create 里一并登记,而不是在 _get 里 —— active_provider()
        # 会绕过 _get 直接建会话,若只在 _get 登记,之后 _run 取输入名就是 KeyError
        self._inames[key] = sess.get_inputs()[0].name
        return sess

    def _get(self, key: str) -> ort.InferenceSession:
        s = self._sess.get(key)
        if s is None:
            s = self._create(key)
            self._sess[key] = s
        return s

    def _run(self, key: str, x: np.ndarray) -> np.ndarray:
        s = self._get(key)
        return s.run(None, {self._inames[key]: x})[0]

    # ------------------------------------------------------------------
    # 前向(与 OVEngine 同名同形状)
    # ------------------------------------------------------------------
    def patcher(self, x: np.ndarray) -> np.ndarray:
        """(1,3,240,240) → (1,226,896) ln_pre 后 tokens。"""
        return self._run("patcher", np.ascontiguousarray(x, dtype=np.float32))

    def tower_full(self, toks: np.ndarray) -> np.ndarray:
        """(1,226,896) → (1,226,640) 整图逐 token 640 空间(已 l2)。
        CLS = out[0, 0:1],patch 特征 = out[0, 1:]。"""
        return self._run("full", np.ascontiguousarray(toks, dtype=np.float32))

    def tower_w(self, seq: np.ndarray) -> np.ndarray:
        """(B,5|10,896) → (B,5|10,640);按序列长度分发 w5/w10,batch 动态。"""
        L = seq.shape[1]
        key = _WIN_KEY.get(L)
        if key is None:
            raise ValueError(f"窗口序列长度只能是 5 或 10,收到 {L}")
        return self._run(key, np.ascontiguousarray(seq, dtype=np.float32))

    # ------------------------------------------------------------------
    def model_sizes_mb(self) -> dict[str, float]:
        """ONNX 文件体积清单(README 记录用)。"""
        return {name: round((self.dir / fn).stat().st_size / 1e6, 1)
                for name, fn in ONNX_FILES.items()}
