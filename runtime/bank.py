"""显式正常库 + A/B 打分臂 —— 差异全部收在 `reduce_sim()` 一个函数里。

## 这个模块解决什么

现行 `_few_token_score(cur, mem) = 0.5·(1 − max cos)`(见 `runtime/pipeline.py:92-96`)
把库摊平成 (M,640),查询的每个 patch 跟**库里所有 patch**比,取最像的那个。
这就是本模块里的 **B 线**(全局最近邻)。

另一端的 **A 线**(位置对应)是:查询的第 (i,j) 格,只跟库里每张图的第 (i,j) 格比。
它多用一个先验 —— "工件每次摆放位置一样"。PaDiM(ICPR 2020)是这条线的代表,
RegAD(ECCV 2022)指出它的前提是**空间刚性**,不成立时不是"效果差"而是**不成立**。

两条线统一写成:

    score(p) = 0.5·(1 − agg_{p' : dist(p,p') ≤ r} max_k cos(q_p , bank_{p',k}))

        r = 0   → A(只比同位置)
        r = 14  → B(比全库;15×15 网格上 Chebyshev 最大距离恰为 14)

⇒ **A 与 B 不是两个方法,是同一个检索式的两个端点**,`BankCfg.radius` 是唯一的开关。
中间值 r ∈ {1,2,3,7} 给出位置先验的连续谱。

## 库的形状

库按**位置**组织:(P, K, 640) —— P 个位置,每个位置存 K 张参考图的特征。
现行实现把它摊平成 (P·K, 640),本模块保持结构化,因为 A 线需要知道"哪条是哪个位置"。

    patch 尺度  P=225 (15×15)      large 3×3  P=169 (13×13)     mid 2×2  P=196 (14×14)
    窗口的位置 = 它左上角 patch 的网格坐标(`win_idx_k*` 的第 0 列)

距离用 **Chebyshev**(方形邻域),不用欧氏 —— 因为欧氏下 15×15 网格的最大距离是
√(14²+14²)≈19.8,`r=14` 覆盖不到角上,自检门"r=14 必须精确退化为 B"就不成立了。

## 两段式 API:为什么要把「算 sim」和「用 sim」拆开

`sim_tensor()` 是唯一昂贵的一步(O(Q·P·K·D));`reduce_sim()` 只是对它的后处理
(O(Q·P·K))。实验里要对同一个库扫十几条臂,差别全在 `reduce_sim` 的配置上 ——
**sim 只算一次,所有臂共用**,整个扫描快一个数量级。

更进一步:参考图子集(G_1/G_4/G_16…)不必重建库再重算 sim —— 全量 sim 的形状是
(Q,P,K),沿 K 轴切片再 reduce,与"先建子库再算"在数学上完全一致(取 max 是逐元素的)。
`scripts/exp/bank_arms.py` 的两道自检门就是在验这件事。

## 自检门

`radius=14` 时 `reduce_sim()` 必须与 `_few_token_score()` **逐位一致**(<1e-6)。
对不上 ⇒ A/B 隔离点写错了。见 `scripts/exp/bank_arms.py --selftest`。

## 措辞纪律

这是"把两个已有实现放进同一实现、用一个半径参数复现两端",**不是融合创新**。
出处:PaDiM [arXiv:2011.08785] / PatchCore [arXiv:2106.08265] / MuSc [arXiv:2401.16753]
(仅借聚合统计量;MuSc 的 MSM 是 transductive 的,本项目从未实现,也不照搬)。

运行时零 torch(仅 numpy),`pipeline.py` 与 `winclip.py` 双向复用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# 三个尺度的位置几何。patch 尺度按 15×15 铺;窗口尺度用其左上角 patch 的坐标。
GRID = 15
SCALES = {
    #  name        P    边长
    "patch": (GRID * GRID, GRID),
    "large": (13 * 13, 13),
    "mid": (14 * 14, 14),
}
# 尺度 → 特征缓存里的键(`/tmp/feat_cache_good/<cls>/*.npz`)
CACHE_KEY = {"patch": "full", "large": "w3", "mid": "w5"}

# 15×15 网格上 Chebyshev 最大距离 = 14 ⇒ r=14 覆盖全球,精确退化为 B
R_FULL = 14


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class BankCfg:
    """库的构造与检索配置。所有可扫的臂都在这一个对象上。"""
    radius: int = R_FULL
    """A/B 唯一开关。0=A(同位置),14=B(全局最近邻),中间=位置先验的连续谱。"""
    agg: str = "max"
    """在半径内对"各位置的最优 cos"做聚合:max | topk | interval。"""
    topk: int = 5
    """agg='topk' 时的 j;agg='interval' 时不用。"""
    reweight: bool = False
    """PatchCore 密度重加权:库整体附近都常见的匹配降权。"""
    core_ratio: float | None = None
    """覆盖率核心集(PatchCore 贪心 k-center)保留比例;None = 不净化。"""
    novelty_tau: float | None = None
    """新颖性门控阈值(本项目已复现的 2608.17775 配方);None = 不过滤。"""

    def __post_init__(self):
        if self.agg not in ("max", "topk", "interval"):
            raise ValueError(f"agg 只能是 max/topk/interval,收到 {self.agg!r}")
        if self.radius < 0:
            raise ValueError("radius 必须 ≥ 0(0=A,14=B)")
        if self.topk < 1:
            raise ValueError("topk 必须 ≥ 1")
        if self.core_ratio is not None and not 0.0 < self.core_ratio <= 1.0:
            raise ValueError("core_ratio 取 (0,1]")

    @property
    def is_global(self) -> bool:
        """radius 是否已大到覆盖全球 ⇒ 该尺度退化为 B。"""
        return self.radius >= R_FULL


# ----------------------------------------------------------------------
# 位置几何
# ----------------------------------------------------------------------
def position_rc(scale: str, win_idx: np.ndarray | None,
                grid: int | None = None) -> np.ndarray:
    """某尺度每个条目的网格坐标 (P,2)。

    patch 尺度:`win_idx` 无关,直接按 GRID 铺。
    窗口尺度:`win_idx` 是 (P, k²) 的 patch 下标表,取第 0 列(左上角)定坐标。
    """
    if scale == "patch":
        p = GRID * GRID
        return np.stack([np.arange(p) // GRID, np.arange(p) % GRID], axis=1)
    first = np.asarray(win_idx)[:, 0] - 1          # 缓存里 win_idx 是 1-based
    g = int(round(np.sqrt(first.max() + 1))) if grid is None else grid
    g = GRID if g * g < first.max() + 1 else g
    return np.stack([first // GRID, first % GRID], axis=1)


def cheb_within(rc: np.ndarray, q_idx: np.ndarray, radius: int) -> np.ndarray:
    """(Q, P) 布尔掩码:查询条目 q 与库条目 p 的 Chebyshev 距离是否 ≤ radius。

    半径 ≥ R_FULL 时直接返回全 True(省掉一次 O(QP) 计算,也让 r=14 的
    退化路径与 `_few_token_score` 走的代码完全一致)。
    """
    P = rc.shape[0]
    if radius >= R_FULL:
        return np.ones((q_idx.shape[0], P), dtype=bool)
    q = rc[q_idx]                                   # (Q,2)
    d = np.maximum(np.abs(q[:, None, 0] - rc[None, :, 0]),
                   np.abs(q[:, None, 1] - rc[None, :, 1]))   # Chebyshev
    return d <= radius


# ----------------------------------------------------------------------
# 库
# ----------------------------------------------------------------------
class Bank:
    """按位置组织的正常库。

    feats[s] : (P, K, 640) float32,L2 归一化。第 0 维是位置,第 1 维是参考图。
    rc[s]    : (P, 2) int64,网格坐标。
    """

    def __init__(self, feats: dict[str, np.ndarray], rc: dict[str, np.ndarray],
                 cfg: BankCfg, n_img: int, meta: dict | None = None):
        self.feats = feats
        self.rc = rc
        self.cfg = cfg
        self.n_img = int(n_img)
        self.meta = meta or {}
        self._mask_cache: dict[tuple[int, int, bytes], np.ndarray] = {}
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        for s, a in self.feats.items():
            if s not in SCALES:
                raise ValueError(f"未知尺度 {s!r};只认 {list(SCALES)}")
            if a.ndim != 3:
                raise ValueError(f"{s}: 期望 (P,K,640),收到 {a.shape}")
            if a.shape[0] != self.rc[s].shape[0]:
                raise ValueError(f"{s}: P={a.shape[0]} 与 rc={self.rc[s].shape[0]} 不一致")
            if a.shape[0] != SCALES[s][0]:
                raise ValueError(f"{s}: 位置数 {a.shape[0]} ≠ 约定 {SCALES[s][0]}")
            n = np.linalg.norm(a, axis=-1)
            if not np.allclose(n, 1.0, atol=1e-3):
                raise ValueError(
                    f"{s}: 特征未 L2 归一化(范数 {n.min():.4f}~{n.max():.4f}),"
                    f"cos 不成立 —— 先用 l2norm() 处理")

    # ------------------------------------------------------------------
    # sim 张量:算一次,供所有臂复用
    # ------------------------------------------------------------------
    def sim_tensor(self, query: np.ndarray, scale: str) -> np.ndarray:
        """(Q,640) 查询 → (Q, P, K) 余弦张量。

        **这是唯一昂贵的一步**(O(Q·P·K·D))。`radius` / `agg` / `reweight`
        都只是对它的后处理,所以实验里对同一张查询图只算一次、所有臂共用 ——
        否则每个臂重算一遍,整个扫描会慢一个数量级。
        """
        bank_p = self.feats[scale]                       # (P,K,D)
        P, K, D = bank_p.shape
        flat = bank_p.reshape(P * K, D)
        return (query @ flat.T).reshape(query.shape[0], P, K)

    def token_score(self, query: np.ndarray, scale: str,
                    q_pos: np.ndarray | None = None) -> np.ndarray:
        """(Q,640) 查询 × 本库的该尺度 → (Q,) 每个查询条目的异常分。

        便捷封装(`sim_tensor` + `reduce_sim`)。要扫多条臂时请直接用那两个,
        别在循环里调本函数 —— 会重复算 sim。
        """
        sim = self.sim_tensor(query, scale)
        return reduce_sim(sim, self.rc[scale], self.cfg, q_pos, self._mask_cache)

    # ------------------------------------------------------------------
    # 净化
    # ------------------------------------------------------------------
    def coverage_curve(self) -> np.ndarray:
        """按 k-center 贪心顺序,库逐步扩张时的覆盖率曲线 —— 净化取核心集的依据。

        返回 (K,) 覆盖率(0~1):第 j 项 = 前 j 张参考图在特征空间的覆盖。
        覆盖率定义:每个候选点到已选集最近距离,取中位数相对全库中位数的下降比例。
        纯诊断量,不参与打分。
        """
        # 用 patch 尺度做代表:它是三尺度里唯一逐位置最细的
        a = self.feats["patch"]                          # (P,K,D)
        P, K, D = a.shape
        flat = a.transpose(1, 0, 2).reshape(K, P * D)    # (K, P*D) 每张图一行
        sel = [0]
        d = np.linalg.norm(flat - flat[0], axis=1)
        out = []
        for _ in range(K):
            out.append(float(np.median(d[sel].min(axis=0))))
            j = int(np.argmax(d))
            if j in sel:
                break
            sel.append(j)
            d = np.minimum(d, np.linalg.norm(flat - flat[j], axis=1))
        out = np.asarray(out, dtype=np.float64)
        return 1.0 - out / max(out[0], 1e-12)

    def prune(self, keep_idx: np.ndarray, cfg: BankCfg | None = None) -> "Bank":
        """按参考图下标保留子集 → 新 Bank(净化后的库)。

        注意:扫臂时**不必**用它 —— `sim_tensor` 沿 K 轴切片等价且更省。
        它是给"要落盘的净化后库"用的。
        """
        idx = np.asarray(keep_idx)
        return Bank({s: a[:, idx, :].copy() for s, a in self.feats.items()},
                    {s: r.copy() for s, r in self.rc.items()},
                    cfg or self.cfg, len(idx), {**self.meta, "pruned_from": self.n_img})

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p / "bank.npz",
            **{f"feat_{s}": a for s, a in self.feats.items()},
            **{f"rc_{s}": r for s, r in self.rc.items()},
            n_img=np.int64(self.n_img))
        (p / "cfg.json").write_text(
            json.dumps({"cfg": asdict(self.cfg), "meta": self.meta},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Bank":
        p = Path(path)
        z = np.load(p / "bank.npz")
        d = json.loads((p / "cfg.json").read_text(encoding="utf-8"))
        feats = {s: z[f"feat_{s}"] for s in SCALES if f"feat_{s}" in z}
        rc = {s: z[f"rc_{s}"] for s in SCALES if f"rc_{s}" in z}
        return cls(feats, rc, BankCfg(**d["cfg"]), int(z["n_img"]), d.get("meta", {}))


# ----------------------------------------------------------------------
# 打分:A/B 唯一开关
# ----------------------------------------------------------------------
def reduce_sim(sim: np.ndarray, rc: np.ndarray, cfg: BankCfg,
               q_idx: np.ndarray | None = None,
               mask_cache: dict | None = None) -> np.ndarray:
    """(Q,P,K) 余弦张量 + 配置 → (Q,) 异常分。**A/B 唯一开关就在这里。**

    三步,顺序固定:
      1. 对 K 取 max          —— 选最像的那张参考图(与现行 `_few_token_score` 一致)
      2. 半径内取最优         —— radius 在这里生效:r=0 只看同位置,r=14 看全库
      3. agg                  —— 在半径内对"各位置的最优 cos"做 max / topk / interval
    """
    Q, P, _K = sim.shape
    q_idx = np.arange(P) if q_idx is None else np.asarray(q_idx)
    if q_idx.shape[0] != Q:
        raise ValueError(f"q_pos 长度 {q_idx.shape[0]} ≠ 查询行数 {Q}")

    best = sim.max(axis=2)                               # (Q,P) 选参考图

    if cfg.is_global:
        # 与 _few_token_score 同路径:全库取最优,不动掩码
        b = best
    else:
        if mask_cache is not None:
            key = (cfg.radius, P, q_idx.tobytes())
            m = mask_cache.get(key)
            if m is None:
                m = cheb_within(rc, q_idx, cfg.radius)
                mask_cache[key] = m
        else:
            m = cheb_within(rc, q_idx, cfg.radius)
        b = np.where(m, best, -np.inf)

    b = b.max(axis=1)                                    # 半径内取最优位置
    # 半径内一个位置都没有 ⇒ 该查询条目无参考。合法输入下不会发生(r≥0 时至少
    # 包含自身),但要显式报错而不是静默出 -inf。**必须检查在 max 之后** ——
    # 半径外的 -inf 是预期的,只有整行全 -inf 才是错。
    if not np.all(np.isfinite(b)):
        raise RuntimeError(f"{int((~np.isfinite(b)).sum())} 个查询条目在 "
                           f"radius={cfg.radius} 内没有任何库位置")
    if cfg.reweight:
        b = _reweight(b, np.where(np.isfinite(best), best, -np.inf), cfg)
    return 0.5 * (1.0 - _agg(b, cfg))


def _reweight(best: np.ndarray, sim_pos: np.ndarray, cfg: BankCfg) -> np.ndarray:
    """PatchCore 密度重加权:若最优匹配**之外**还有位置同样很像,说明该匹配不独特、
    降权;只有它自己像、别处都不像,才是真异常。用"除去最优后的次优"作密度代理
    —— 不需要额外存库外样本。"""
    if sim_pos.shape[1] < 2:
        return best
    second = np.sort(sim_pos, axis=1)[:, -2]
    w = np.clip(1.0 - second / np.maximum(np.abs(best), 1e-6), 0.0, 1.0)
    return best * (0.5 + 0.5 * w)


def _agg(best: np.ndarray, cfg: BankCfg) -> np.ndarray:
    """(Q,) cos → (Q,) 聚合值。"""
    if cfg.agg == "max":
        return best
    if cfg.agg == "topk":
        j = min(cfg.topk, best.shape[-1])
        return np.sort(best, axis=-1)[..., -j:].mean(axis=-1)
    # interval:稳健区间平均(MuSc 借用的聚合统计量)
    lo = np.quantile(best, 0.25, axis=-1, keepdims=True)
    hi = np.quantile(best, 0.75, axis=-1, keepdims=True)
    inside = (best >= lo) & (best <= hi)
    v = np.where(inside, best, np.nan)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(v, axis=-1)
    return np.where(np.isfinite(v), v, best.max(axis=-1))


# ----------------------------------------------------------------------
# 构造
# ----------------------------------------------------------------------
def l2norm(a: np.ndarray) -> np.ndarray:
    return (a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
            ).astype(np.float32)


def build_bank(feats_by_img: dict[str, np.ndarray],
               win_idx: dict[str, np.ndarray],
               cfg: BankCfg | None = None,
               keep_idx: np.ndarray | None = None,
               meta: dict | None = None) -> Bank:
    """(K 张图的三尺度特征) → 按位置组织的 Bank。

    feats_by_img[s] : (K, P, 640) —— **图片序**,与缓存 npz 同构,先 L2 再转置。
    win_idx[s]      : 窗口尺度用 (P, k²) 的 patch 下标表;patch 尺度传 None。
    keep_idx        : 只保留这些参考图(净化);None = 全用。
    """
    cfg = cfg or BankCfg()
    feats, rc = {}, {}
    for s in SCALES:
        a = np.asarray(feats_by_img[s], dtype=np.float32)
        if keep_idx is not None:
            a = a[np.asarray(keep_idx)]
        if a.shape[0] == 0:
            raise ValueError(f"{s}: 库为空(keep_idx 全被滤掉了?)")
        if a.shape[1] != SCALES[s][0]:
            raise ValueError(f"{s}: 位置数 {a.shape[1]} ≠ 约定 {SCALES[s][0]}")
        a = l2norm(a)
        feats[s] = np.ascontiguousarray(a.transpose(1, 0, 2))    # (P,K,640)
        rc[s] = position_rc(s, None if win_idx is None else win_idx.get(s))
    return Bank(feats, rc, cfg, feats["patch"].shape[1], meta)


def load_cache_features(cache_dir: str | Path, cls: str,
                        n_max: int | None = None) -> dict[str, np.ndarray]:
    """从特征缓存目录读一个类的全部参考图 → {scale: (K,P,640)}。

    缓存键(`full`/`w3`/`w5`)见 `CACHE_KEY`;`full` 的第 0 行是 CLS token,跳过。
    文件按名排序后取前 `n_max` 张(None = 全取)。
    """
    d = Path(cache_dir) / cls
    fs = sorted(d.glob("[0-9]*.npz"))
    if not fs:
        raise FileNotFoundError(f"{d} 里没有 [0-9]*.npz")
    if n_max is not None:
        fs = fs[:n_max]
    out: dict[str, list] = {s: [] for s in SCALES}
    for f in fs:
        z = np.load(f)
        out["patch"].append(z["full"][1:].astype(np.float32))
        out["large"].append(z["w3"].astype(np.float32))
        out["mid"].append(z["w5"].astype(np.float32))
    return {s: np.stack(v) for s, v in out.items()}
