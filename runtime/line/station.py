"""工位运行时:把传统前置、级联检测、判定、I/O 串成一条产线。

完整数据流(每件货一次):
    触发 → 取帧 → [传统 CV] 光照归一化 → 定位/规整 → 工件裁切 → 可疑度+ROI
         → [模型] CLIP 归一化 → 整图粗筛 → (早退?) → 窗口精检 → few-shot
         → [后处理] 上采样 → 拓扑过滤 → 面积过滤 → 判定
         → [I/O] 剔除信号 + 结果落盘 + 可视化

I/O 抽象的意义:产线设备千差万别(相机厂商、PLC 协议、剔除机构),
把触发/输出抽成接口,算法侧只认"进来一帧、出去一个判定",换设备不动算法。
本仓库实现 sequence(离线复跑)与 camera(USB/V4L2)两种,PLC 留接口。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..ov_engine import OVEngine
from ..pipeline import GRID, OVPipeline
from .cascade import CascadeDetector
from .config import LineConfig
from .localizer import WorkpieceLocalizer, build_roi_weight
from .metrics import LineReporter
from .preprocess_cv import normalize_illumination, patch_suspicion, suspicion_map
from .scheduler import CostModel, TaktScheduler
from .verdict import decide, postprocess, render_overlay


# ----------------------------------------------------------------------
# 触发与输出抽象
# ----------------------------------------------------------------------
class FrameSource:
    """取帧接口。sequence = 按文件列表;camera = 设备实时取流。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cap = None

    def frames(self, frame_dir: str | Path | None = None):
        if self.cfg.trigger == "sequence":
            root = Path(frame_dir or self.cfg.frame_source)
            files = sorted(p for p in root.rglob("*")
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
            for p in files:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    yield p.name, g
        else:
            self._cap = cv2.VideoCapture(
                int(self.cfg.frame_source) if str(self.cfg.frame_source).isdigit()
                else self.cfg.frame_source)
            i = 0
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield f"frame_{i:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                i += 1
            self._cap.release()


class RejectOutput:
    """剔除信号接口。log 仅记录;gpio/modbus 为现场对接预留。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.events: list[dict] = []

    def send(self, frame_id: str, verdict) -> None:
        if verdict.action in ("NG", "UNKNOWN"):
            ev = {"ts": time.time(), "frame": frame_id, "action": verdict.action,
                  "n_defects": verdict.n_defects, "score": verdict.image_score}
            self.events.append(ev)
            if self.cfg.reject_output == "log":
                print(f"  [REJECT] {frame_id} {verdict.action} "
                      f"score={verdict.image_score:.3f} defects={verdict.n_defects}",
                      flush=True)
            # gpio / modbus 分支:现场按 PLC 协议实现脉冲输出与握手


# ----------------------------------------------------------------------
@dataclass
class StationState:
    """工位状态:产线调试时第一眼要看的东西。"""
    frames: int = 0
    ng: int = 0
    rework: int = 0
    unknown: int = 0
    locate_fail: int = 0
    started_at: float = field(default_factory=time.time)

    def update(self, v) -> None:
        self.frames += 1
        if v.action == "NG":
            self.ng += 1
        elif v.action == "REWORK":
            self.rework += 1
        elif v.action == "UNKNOWN":
            self.unknown += 1


class StationRuntime:
    """单工位运行时。负责装配与编排,不含算法细节。"""

    def __init__(self, cfg: LineConfig, engine: OVEngine | None = None):
        self.cfg = cfg
        cfg.validate()
        self.engine = engine or OVEngine(cfg.deploy_dir, cfg.device)
        self.pipe = OVPipeline(self.engine, cfg.text_dir)
        self.pipe.set_class(cfg.class_name.replace("_", " "))

        # 传统前置
        self.flatfield = self._load_flatfield()
        self.localizer = self._load_localizer()
        self.roi_w_240 = build_roi_weight(cfg.roi, 240)
        self.roi_w_patch = self._downsample_roi(self.roi_w_240)

        # 节拍调度
        cost = CostModel.load(cfg.takt.cost_model_path)
        if cfg.takt.ms_per_token > 0:
            cost = CostModel(cfg.takt.ms_per_token, cost.fixed_ms, "config")
        self.cost = cost
        self.sched = TaktScheduler(cfg.takt.takt_ms, cfg.takt.safety_margin_ms,
                                   cost, cfg.takt.min_windows,
                                   cfg.takt.overrun_policy)

        # 级联
        self.det = CascadeDetector(self.pipe, self.sched, cfg.cascade,
                                   cfg.cv, cfg.roi, self.flatfield)
        self.det.set_roi_patch_weight(self.roi_w_patch)
        self._load_guard()

        self.state = StationState()
        self.reporter = LineReporter(cfg.station_id, cfg.takt.takt_ms,
                                     cfg.fingerprint())
        self.output = RejectOutput(cfg.io)

    # ------------------------------------------------------------------
    def _load_flatfield(self):
        if self.cfg.cv.illum_norm == "flatfield" and self.cfg.cv.flatfield_path:
            p = Path(self.cfg.cv.flatfield_path)
            if p.exists():
                return np.load(p)["flatfield"].astype(np.float32)
            raise FileNotFoundError(f"平场校正文件不存在: {p}")
        return None

    def _load_localizer(self):
        from .config import GeometryCfg
        if self.cfg.geometry.mode == "pre_cropped":
            return WorkpieceLocalizer(self.cfg.geometry, self.cfg.cv)
        p = Path(self.cfg.geometry.template_path)
        if not p.exists():
            raise FileNotFoundError(f"定位模板不存在: {p}(先跑 line_calib)")
        return WorkpieceLocalizer.from_file(p, self.cfg.geometry, self.cfg.cv)

    def _load_guard(self):
        """良品标定护栏;缺失时关闭早退(宁慢不误),而不是用默认值硬跑。"""
        gp = Path(self.cfg.cascade.early_exit_guard_path) \
            if self.cfg.cascade.early_exit_guard_path else None
        if gp and gp.exists():
            self.det.set_guard(json.loads(gp.read_text(encoding="utf-8")))
            self.guard = json.loads(gp.read_text(encoding="utf-8"))
        else:
            if self.cfg.cascade.early_exit_enable:
                print("[warn] 未找到良品标定护栏文件,自动关闭早退"
                      "(避免用拍脑袋阈值误放不良品)", flush=True)
            self.cfg.cascade.early_exit_enable = False
            self.guard = None

    @staticmethod
    def _downsample_roi(roi240: np.ndarray) -> np.ndarray:
        return cv2.resize(roi240, (GRID, GRID), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    def prepare_input(self, gray: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """场景灰度 → (模型输入 (1,3,240,240), 传统产物, 诊断信息)。

        这一步是"传统 CV 与模型的分界线",也是最容易出错的地方:
        进入模型的图必须与 research 路径的预处理**同源同尺度**,
        否则精度对不上,而且很难查(表现为"产线比离线低几个点")。
        """
        t0 = time.perf_counter()
        diag = {"locate_ok": True, "locate_score": 1.0, "rotation_deg": 0.0}

        # 1) 定位/规整
        loc = self.localizer.locate(gray)
        diag.update(locate_ok=loc.ok, locate_score=loc.score,
                    rotation_deg=loc.rotation_deg, locate_reason=loc.reason)
        std = self.localizer.warp_to_standard(gray, loc)
        crop = self.localizer.workpiece_crop(std)

        # 2) 光照归一化(在工件图上做,背景不参与)
        norm = normalize_illumination(crop, self.cfg.cv, self.flatfield)

        # 3) 可疑度图 + ROI 权重(零神经算力)
        susp = suspicion_map(norm, self.cfg.cv)
        susp_patch = patch_suspicion(susp, GRID)

        # 4) 模型输入:统一到 240 → CLIP 归一化(与 OVEngine.preprocess_rgb 同式)
        x = self._to_model_input(norm)
        cv_ms = (time.perf_counter() - t0) * 1e3
        return x, {"susp_patch": susp_patch, "roi_patch": self.roi_w_patch,
                   "cv_ms": cv_ms}, diag

    def _to_model_input(self, gray_norm: np.ndarray) -> np.ndarray:
        """归一化灰度 → (1,3,240,240),逐位复用 OVEngine 的变换常量。"""
        g = gray_norm.astype(np.float32)
        if g.shape != (240, 240):
            g = cv2.resize(g, (240, 240), interpolation=cv2.INTER_LINEAR)
        rgb = np.clip(g, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)
        return OVEngine.preprocess_rgb(rgb)

    # ------------------------------------------------------------------
    def process_frame(self, frame_id: str, gray: np.ndarray,
                      is_defect: bool | None = None):
        """单帧全流程 → (verdict, 诊断字典)。"""
        t0 = time.perf_counter()
        x, cv_maps, diag = self.prepare_input(gray)

        res = self.det.process(x, cv_maps)

        # 后处理与判定
        if res.degraded and res.map_patch is None:
            verdict = decide(res.image_score, [], self.cfg.verdict,
                             degraded=True, locate_ok=diag["locate_ok"])
        else:
            binm, defects = postprocess(res.map_patch, self.cfg.verdict,
                                        self.cfg.geometry.pixel_size_mm)
            verdict = decide(res.image_score, defects, self.cfg.verdict,
                             degraded=res.degraded,
                             locate_ok=diag["locate_ok"])
            verdict.map_bin = binm
        verdict.degraded = res.degraded or not diag["locate_ok"]

        ct_ms = (time.perf_counter() - t0) * 1e3
        self.state.update(verdict)
        self.output.send(frame_id, verdict)

        stages = dict(res.stages_ms)
        stages["cv_ms"] = round(cv_maps["cv_ms"], 2)
        stages["total_ms"] = round(ct_ms, 2)
        self.reporter.add_frame(
            verdict, ct_ms, stages, res.n_win3, res.n_win2, res.tokens,
            res.early_exit, res.degraded, is_defect)

        diag.update(res.stages_ms, degraded=res.degraded,
                    degrade_reason=res.degrade_reason,
                    tokens=res.tokens, n_win3=res.n_win3, n_win2=res.n_win2,
                    early_exit=res.early_exit, ct_ms=round(ct_ms, 2),
                    image_score=round(res.image_score, 4))
        return verdict, diag, res

    # ------------------------------------------------------------------
    def run(self, frame_dir: str | Path | None = None,
            save_overlay_dir: str | Path | None = None,
            max_frames: int = 0, fps_limit: float | None = None) -> dict:
        """跑一批帧(离线复跑/离线评估用),返回产线报表。"""
        src = FrameSource(self.cfg.io)
        fps = self.cfg.io.fps_limit if fps_limit is None else fps_limit
        overlay_dir = Path(save_overlay_dir) if save_overlay_dir else None
        if overlay_dir:
            overlay_dir.mkdir(parents=True, exist_ok=True)

        for i, (fid, gray) in enumerate(src.frames(frame_dir)):
            if max_frames and i >= max_frames:
                break
            t_f = time.perf_counter()
            verdict, diag, res = self.process_frame(fid, gray)
            if overlay_dir and res.map_patch is not None:
                ov = render_overlay(gray, res.map_patch, verdict)
                cv2.imwrite(str(overlay_dir / f"{Path(fid).stem}_ov.jpg"), ov)
            if fps > 0:
                slack = 1.0 / fps - (time.perf_counter() - t_f)
                if slack > 0:
                    time.sleep(slack)      # 模拟产线节拍的强制间隔

        rep = self.reporter.report()
        out_dir = Path(self.cfg.io.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = out_dir / f"{self.cfg.station_id}_{ts}.json"
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        rep["_report_path"] = str(p)
        return rep
