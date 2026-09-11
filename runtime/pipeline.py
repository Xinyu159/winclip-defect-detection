"""OVPipeline:镜像 winclip.WinCLIP 打分结构,numpy 实现,可逐段对拍。

与 winclip.py 语义逐位一致的路径(整图全量窗口):
    map   = 三路调和 3/(1/m48 + 1/m32 + 1/cls)          (每尺度各自由覆盖窗口调和到 patch)
    few   = few 三尺度平均(窗口两尺度调和 + patch 尺度),map += few
    image = zero: cls 异常概率;few:(cls + max(few map))/2

window_subset 扩展(混合流水线用,传统 CV 前置只算覆盖窗口):
    每尺度只对"已算窗口"做调和;patch p 在尺度 s 缺席(无已算窗口覆盖)时,
    m[p] = N(p) / Σ_{i∈S(p)} 1/m_i[p],S(p) = 覆盖 p 的已算尺度 ∪ {cls},
    N(p)=|S(p)|。全量时 N≡3,与原式逐位一致;CLS 恒在保证 map 处处有值,
    不引入新参数、不碰 GT。few 分支同理按可用尺度计数归一(patch 尺度恒可得)。

运行时零 torch/open_clip(仅 numpy);文本原型由 build_text_protos 离线预计算,
set_class 只读 npz,不做任何文本编码。

引擎只依赖 EngineBase 接口(patcher/tower_full/tower_w/window_indices),
**不 import 具体后端** —— 否则本模块会被绑死在 OpenVINO 上,ONNX/ORT 侧
无法复用同一套打分(pipeline 是"算法正确性的唯一真相",必须后端无关)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .engine_base import EngineBase

GRID = 15
N_PATCH = GRID * GRID


class OVPipeline:
    def __init__(self, engine: EngineBase, text_dir: str | Path):
        self.engine = engine
        self.text_dir = Path(text_dir)
        self.pos = None          # normal 原型 (1,640) l2
        self.neg = None          # abnormal 原型 (1,640) l2
        self.temp = None
        self.gallery = None      # few-shot 参考 {large/mid/patch: (N,640)}

    # ------------------------------------------------------------------
    # 类设置与 few-shot 参考库
    # ------------------------------------------------------------------
    def set_class(self, cls_name: str) -> None:
        """cls_name 接受**同一套调用约定**下的类名("metal_nut" 或 "metal nut")。

        文本原型文件名用下划线(build_text_protos 按 MVTec 类名落盘),
        而 CLIP 提示词里物体名用空格。早期实现要求调用方自己决定传哪种,
        导致 evaluate/winclip 侧传空格、文件侧找 "metal nut.npz" 必然失败
        (MVTec 15 类里只有 metal_nut 带下划线,问题被掩盖到最后一刻)。

        这里统一:文件名按下划线,空格/下划线输入都接受。
        """
        self.cls_name = cls_name
        fname = cls_name.replace(" ", "_") + ".npz"
        path = self.text_dir / fname
        if not path.exists():
            avail = sorted(p.stem for p in self.text_dir.glob("*.npz"))
            raise FileNotFoundError(
                f"文本原型缺失: {path}\n"
                f"  可用: {', '.join(avail) if avail else '(目录为空)'}\n"
                f"  → 先生成: python scripts/build_text_protos.py --classes all")
        d = np.load(path)
        self.pos, self.neg = d["normal"], d["abnormal"]
        self.temp = float(d["temp"])

    def set_gallery(self, imgs240: np.ndarray) -> None:
        """(k,3,240,240) 已归一化正常图 → 三尺度 640 空间特征(与 torch 版同构)。"""
        g = {"large": [], "mid": [], "patch": []}
        for i in range(imgs240.shape[0]):
            toks = self.engine.patcher(imgs240[i:i + 1])     # (1,226,896)
            full = self.engine.tower_full(toks)              # (1,226,640)
            g["patch"].append(full[0, 1:])                   # (225,640)
            for name, k in (("large", 3), ("mid", 2)):
                ws = self._window_feats(toks, k, None)       # (n_win,640) 全量
                g[name].append(ws)
        self.gallery = {name: np.concatenate(v, axis=0) for name, v in g.items()}

    # ------------------------------------------------------------------
    # 静态打分工具(numpy 版与 torch 版一一对应)
    # ------------------------------------------------------------------
    @staticmethod
    def _prob(feats: np.ndarray, pos: np.ndarray, neg: np.ndarray,
              temp: float) -> np.ndarray:
        """(n,640) × [pos,neg] → 温度 softmax → 异常概率 (n,)。"""
        logits = np.concatenate([feats @ pos.T, feats @ neg.T], axis=1) * temp
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e[:, 1] / e.sum(axis=1)

    @staticmethod
    def _few_token_score(cur: np.ndarray, mem: np.ndarray) -> np.ndarray:
        """(N,640) 查询 × (M,640) 参考 → 每 token 最近邻异常分 0.5·(1−max cos)。"""
        sim = cur @ mem.T
        return 0.5 * (1.0 - sim.max(axis=-1))

    # ------------------------------------------------------------------
    # 窗口机制
    # ------------------------------------------------------------------
    def _window_feats(self, toks: np.ndarray, k: int, subset=None) -> np.ndarray:
        """窗口子序列 [CLS+窗口 patch] 批量过塔 → 窗口级 640 特征 (n,640)。

        subset: 窗口行号(全量 n_win 中挑),None = 全量。返回行按 subset 顺序。
        """
        idx_all = self.engine.window_indices(k)              # (n_win, k*k)
        chosen = idx_all if subset is None else idx_all[subset]
        seq = np.concatenate(
            [np.zeros((chosen.shape[0], 1), dtype=np.int64), chosen], axis=1)
        rows = toks[0][seq]                                  # (n, k²+1, 896)
        out = self.engine.tower_w(rows)                      # (n, L, 640)
        return out[:, 0]                                     # 窗口 CLS

    @staticmethod
    def _scatter_harmonic(win_prob: np.ndarray, win_idx: np.ndarray):
        """窗口级分数 → 每 patch 调和平均 (map225, cnt225)。

        A[p,w]=w 是否覆盖 p;m[p]=cnt/inv(仅已算窗口);cnt=0 的 patch 记缺席。
        """
        n_win = win_prob.shape[0]
        A = np.zeros((n_win, N_PATCH), dtype=np.float32)
        A[np.arange(n_win)[:, None], win_idx - 1] = 1.0
        cnt = A.sum(axis=0)
        # 调和平均对 0 敏感:某窗口概率恰为 0 时 1/0=inf,整幅 map 变 NaN。
        # softmax 在充分确信时确实会下溢到 0(实测 screw/metal_nut 等类发生),
        # 所以必须夹住——用 1e-12 兜底,对非零概率的结果影响 <1e-9。
        # 注意:这同时说明**窗口精检对"到处都是零"的场景不敏感**,
        # 调和平均的天性会压制小缺陷,是级联在弥散缺陷类上掉点的原因之一。
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = A.T @ (1.0 / np.maximum(win_prob, 1e-12))
            m = np.where(cnt > 0, cnt / inv, 0.0)
        return m.astype(np.float32), cnt.astype(np.int32)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def anomaly_maps(self, img240: np.ndarray, use_few: bool = False,
                     window_subset: dict | None = None,
                     _t: dict | None = None):
        """(1,3,240,240) → (map (15,15), img_score, diag)。

        window_subset = {"w5": 行号数组, "w3": 行号数组}(None = 全量);
        _t 为内部计时字典(调试用),diag 含各阶段耗时 ms 与已算窗口计数。
        """
        if _t is None:
            _t = {}
        t0 = time.perf_counter()
        assert self.pos is not None, "先 set_class()"

        toks = self.engine.patcher(img240)                   # (1,226,896)
        _t["patcher_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        full = self.engine.tower_full(toks)                  # (1,226,640)
        cls_prob = self._prob(full[0, :1], self.pos, self.neg,
                              self.temp)[0]                  # 标量
        _t["full_ms"] = (time.perf_counter() - t1) * 1e3

        # 多尺度窗口(默认 3×3/2×2 全量;subset 时只算指定窗口)。
        # 每尺度窗口特征只算一次,zero 概率与 few 最近邻共用(与 torch 版一致)
        win_feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        patch_maps, cnts = {}, {}
        diag_n = {}
        t2 = time.perf_counter()
        for name, k, key in (("m48", 3, "w3"), ("m32", 2, "w5")):
            sub = None if window_subset is None else window_subset[key]
            diag_n[key] = (169 if k == 3 else 196) if sub is None else len(sub)
            idx = self.engine.window_indices(k)
            chosen = idx if sub is None else idx[sub]        # 行序与 ws 对齐
            ws = self._window_feats(toks, k, sub)            # (n, 640)
            win_feats[key] = (ws, chosen)
            wp = self._prob(ws, self.pos, self.neg, self.temp)
            m, cnt = self._scatter_harmonic(wp, chosen)
            patch_maps[name], cnts[name] = m, cnt
        _t["wins_ms"] = (time.perf_counter() - t2) * 1e3

        # 广义三路调和:CLS 恒在,窗口尺度只在"有已算窗口覆盖"的 patch 计入
        t3 = time.perf_counter()
        inv = np.full(N_PATCH, 1.0 / cls_prob, dtype=np.float32)
        n_terms = np.ones(N_PATCH, dtype=np.float32)         # 恒含 CLS
        for name in ("m48", "m32"):
            m, cnt = patch_maps[name], cnts[name]
            present = cnt > 0
            inv[present] += 1.0 / m[present]
            n_terms[present] += 1.0
        m_all = n_terms / inv                                # (225,)
        _t["mix_ms"] = (time.perf_counter() - t3) * 1e3

        # few-shot:窗口尺度按已算窗口调和,patch 尺度恒可得,可用尺度计数归一
        t4 = time.perf_counter()
        img_score = float(cls_prob)
        if use_few:
            assert self.gallery is not None, "先 set_gallery()"
            g = self.gallery
            patch_shared = full[0, 1:]                       # (225,640)
            patch_score = self._few_token_score(patch_shared, g["patch"])
            few_num = patch_score.copy()
            few_den = np.ones(N_PATCH, dtype=np.float32)
            for name, k, key, gname in (("m48", 3, "w3", "large"),
                                        ("m32", 2, "w5", "mid")):
                ws, chosen = win_feats[key]                  # 复用 zero 分支结果
                tok = self._few_token_score(ws, g[gname])    # (n_win,)
                m, cnt = self._scatter_harmonic(tok, chosen)
                present = cnt > 0
                few_num[present] += m[present]
                few_den[present] += 1.0
            few_map = few_num / few_den
            m_all = m_all + few_map
            img_score = (float(cls_prob) + float(few_map.max())) / 2.0
        _t["few_ms"] = (time.perf_counter() - t4) * 1e3

        diag = {"computed_windows": diag_n,
                "ms": {k: round(v, 2) for k, v in _t.items()},
                "use_few": bool(use_few),
                "subset": window_subset is not None}
        return m_all.reshape(GRID, GRID), img_score, diag

    # ------------------------------------------------------------------
    # 时延分解(基准用)
    # ------------------------------------------------------------------
    def frame_stages_ms(self, img240: np.ndarray, use_few: bool = False,
                        window_subset: dict | None = None) -> dict:
        """单帧逐段耗时:与 anomaly_maps 同路径,只回计时不含图。"""
        _t: dict[str, float] = {}
        self.anomaly_maps(img240, use_few=use_few,
                          window_subset=window_subset, _t=_t)
        return {k: round(v, 2) for k, v in _t.items()}
