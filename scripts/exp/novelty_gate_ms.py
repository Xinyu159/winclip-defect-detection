"""novelty-gated 良品入库 —— **多尺度版**(与生产管线同口径)。

单尺度版(`novelty_gate.py`)的结论是"门控无用",但它只用了 `full` patch token,
而本项目生产管线走的是 **2×2 + 3×3 窗口 + 调和合并**。调和合并带**覆盖计数**,
"加 patch 只能让分数下降"的单调性论证**在这里不自动成立** —— 所以必须单独测。

## 与单尺度版的唯一区别:打分口径

    单尺度: loc = few(full[1:], gal.patch)          → max
    多尺度: loc = few(full[1:], gal.patch)
            for 每个窗口尺度: mf,cf = harm(few(win, gal.scale), win_idx)
                              loc[cf>0] += mf[cf>0]; den[cf>0] += 1
            map = loc/den                            → max

每条 query 图都能拿到一张 (225,) 合并图,再取 max。这与
`shot_scaling.py::few_map_of` **逐行同源**,并由本脚本的 `--verify` 自检。

## 门控在哪一层

**每个尺度各自独立门控**:τ_scale = 该尺度评审池 token 到**该尺度基线库**的
1-NN 距离中位数。理由:三个尺度的 token 分布不同,共用一个 τ 没有依据。
`rnd_matched` 也**逐尺度**匹配 gated 的插入数。

## 硬规则(与单尺度版一致)

  - τ 只来自评审池(纯良品),开跑前定一次;从不碰 held-out、从不碰 GT
  - 插入的图全部是良品池里的图;缺陷图从不入库
  - 每个数字带 n

用法:
    python scripts/exp/novelty_gate_ms.py screw,cable,transistor 4 3
    python scripts/exp/novelty_gate_ms.py screw 4 1 --verify   # 只做同源自检
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                        # noqa: E402
from sklearn.metrics import roc_auc_score                 # noqa: E402

from runtime.pipeline import N_PATCH                      # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from novelty_gate import load, patches                    # noqa: E402

# (npz 键, gallery 键, 窗口索引文件名, τ 的臂名后缀)
SCALES = [('full', 'patch', None), ('w3', 'large', 'win_idx_k3'),
          ('w5', 'mid', 'win_idx_k2')]
GATED_Q = {'gated': 0.50, 'gated_q25': 0.25, 'gated_q75': 0.75}
ARMS = ['none', 'ungated', 'gated', 'gated_q25', 'gated_q75', 'rnd_matched']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes")
    ap.add_argument("k", type=int)
    ap.add_argument("reps", type=int, nargs="?", default=3)
    ap.add_argument("--calib", default="/tmp/feat_cache_good")
    ap.add_argument("--cache", default="/tmp/feat_cache")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="只跑同源自检:本脚本的合并分 == shot_scaling.few_map_of")
    return ap.parse_args()


def tokens(z, key, dep):
    """取某尺度的 token 矩阵(已 L2 归一化,与 _few_token_score 同源)。"""
    if key == 'full':
        return z['full'][1:]                      # (225,640),丢 CLS
    return z[key]                                 # w3 (169,640) / w5 (196,640)


def coverage(dep, win_k):
    """窗口 → patch 的覆盖矩阵 A(n_win,225) 与覆盖计数 cnt(225,)。

    ★ 与 OVPipeline._scatter_harmonic 里的 `A` / `cnt` **同一个东西**,
      且 cnt **与数据无关**(只由窗口索引决定),所以整批图共用一个。
    """
    idx = np.load(Path(dep) / f"{win_k}.npy")
    n_win = idx.shape[0]
    A = np.zeros((n_win, N_PATCH), np.float32)
    A[np.arange(n_win)[:, None], idx - 1] = 1.0
    return A, A.sum(axis=0)


def merged_score(bests, covs):
    """三条 best → 每图合并 map 的 max。与 few_map_of 逐行同源。

    bests: [best_full (n,225), best_large (n,169), best_mid (n,196)]
    covs : [(A3,cnt3), (A2,cnt2)]
    """
    n = bests[0].shape[0]
    loc = (0.5 * (1.0 - bests[0]))                 # (n,225) 尺度1 = full
    den = np.ones((n, N_PATCH), np.float32)
    for d_win, (A, cnt) in zip([0.5 * (1.0 - b) for b in bests[1:]], covs):
        # inv = A.T @ (1/d):对每个 patch,把覆盖它的窗口的 1/d 求和
        #   原实现是 A.T @ (1/max(win_prob,1e-12));这里 d_win 按图批量,故写成
        #   (1/d_win) @ A —— A 是 one-hot,两种写法等价。
        inv = (1.0 / np.maximum(d_win, 1e-12)) @ A          # (n,225)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(cnt > 0, cnt / inv, 0.0)
        prf = cnt > 0
        loc[:, prf] += m[:, prf]
        den[:, prf] += 1.0
    return (loc / den).max(axis=1)


def upd(best, qtok, new):
    """库只增 ⇒ 每 token 的最大余弦单调不减 ⇒ 增量复用。"""
    if new.shape[0] == 0:
        return
    sim = qtok @ new.T
    np.maximum(best.reshape(-1), sim.max(axis=1), out=best.reshape(-1))


def main() -> int:
    a = parse_args()
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    RS = np.random.RandomState(a.seed)
    covs = [coverage(a.deploy, k) for k in ('win_idx_k3', 'win_idx_k2')]

    print(f"novelty-gated 良品入库【多尺度】| k={a.k} | reps={a.reps} | seed={a.seed}")
    print("  打分 = full + 2×2 + 3×3 窗口调和合并后取 max(与生产管线同口径)")
    print("  τ 逐尺度独立,只来自评审池,开跑前定一次;不碰 held-out、不碰 GT")
    print()

    if a.verify:
        return _verify(a, covs, classes[0])

    summary = {}
    for cls in classes:
        calib_all = load(Path(a.calib) / cls)
        n_all = len(calib_all)
        if n_all < 2 * a.k + 4:
            print(f"{cls:13s} 良品池 {n_all} 张,不够,跳过")
            continue
        perm = RS.permutation(n_all)
        h = n_all // 2
        calib = [calib_all[i] for i in perm[:h]]
        evg = [calib_all[i] for i in perm[h:2 * h]]
        bad = [np.load(f) for f in sorted((Path(a.cache) / cls).glob("*.npz"))
               if f.name != "gallery.npz"]
        bad = [z for z in bad if z['gt'].any()]
        if len(evg) < 4 or not bad:
            print(f"{cls:13s} 缓存不全,跳过")
            continue

        qs = [np.concatenate([tokens(z, k, a.deploy) for z in evg + bad])
              .astype(np.float32) for k, _g, _d in SCALES]
        n_img = len(evg) + len(bad)
        # ★ best 的形状是 (图数, **该尺度的每图 token 数**) —— 不是库的 token 数。
        #   早先写成 bank.shape[0] 会让 (n_img,225) 和 (n_img,库) 广播失败。
        ntok = [q.shape[0] // n_img for q in qs]

        res = {arm: [] for arm in ARMS + ['full']}
        rates = {arm: [] for arm in ARMS}
        for _rep in range(a.reps):
            sel = RS.choice(len(calib), size=a.k, replace=False)
            review = [i for i in range(len(calib)) if i not in set(sel)]
            bank0 = [np.concatenate([tokens(calib[i], k, a.deploy)
                                     for i in sel]).astype(np.float32)
                     for k, _g, _d in SCALES]
            # τ 逐尺度:评审池全部 token 到**基线**库的 1-NN 距离分位数
            rp = [np.concatenate([tokens(calib[i], k, a.deploy)
                                  for i in review]).astype(np.float32)
                  for k, _g, _d in SCALES]
            taus = [{n: float(np.quantile(
                0.5 * (1.0 - (r @ b.T).max(axis=1)), q))
                for n, q in GATED_Q.items()} for r, b in zip(rp, bank0)]

            def sc(banks):
                bs = [np.full((n_img, n), -np.inf, np.float32)
                      for n in ntok]
                for bt, q, bk in zip(bs, qs, banks):
                    upd(bt, q, bk)
                return merged_score(bs, covs)

            res['none'].append(_auroc(sc(bank0), len(evg)))
            res['full'].append(_auroc(
                sc([np.concatenate([tokens(z, k, a.deploy) for z in calib])
                    .astype(np.float32) for k, _g, _d in SCALES]), len(evg)))

            order = RS.permutation(len(review))
            for arm in ARMS:
                if arm == 'none':
                    continue
                banks = [b.copy() for b in bank0]
                bs = [np.full((n_img, n), -np.inf, np.float32) for n in ntok]
                for bt, q, bk in zip(bs, qs, banks):
                    upd(bt, q, bk)
                curve = []
                for i in order:
                    new_per_scale = []
                    for si, (k, _g, _d) in enumerate(SCALES):
                        cand = tokens(calib[review[i]], k, a.deploy) \
                            .astype(np.float32)
                        if arm == 'ungated':
                            take = np.ones(len(cand), bool)
                        else:
                            dc = 0.5 * (1.0 - (cand @ banks[si].T).max(axis=1))
                            if arm == 'rnd_matched':
                                ntk = int((dc > taus[si]['gated']).sum())
                                take = np.zeros(len(cand), bool)
                                if ntk:
                                    take[RS.choice(len(cand), ntk,
                                                   replace=False)] = True
                            else:
                                take = dc > taus[si][arm]
                        rates[arm].append(float(take.mean()))
                        new_per_scale.append(cand[take])
                    # ★ 库必须同步增长:论文 Eq.(4) 的判据是 min_{m∈M^(r)} ——
                    #   M^(r) 是**当前**库,不是基线库。早先这里只更新了打分侧
                    #   bs、漏了 banks,于是门控永远拿基线库算候选距离,插入率
                    #   恒定 50.0%/75.0%/25.0%(那个"整齐得不正常"的插入率
                    #   就是破绽);而 AUROC 照常变化,从结果上完全看不出来。
                    for si, nw in enumerate(new_per_scale):
                        if nw.shape[0]:
                            banks[si] = np.concatenate([banks[si], nw])
                        upd(bs[si], qs[si], nw)
                    curve.append(_auroc(merged_score(bs, covs), len(evg)))
                res[arm].append(curve)

        summary[cls] = (res, n_all, len(calib), len(evg), len(bad), len(review))
        print(f"{cls:13s} 良品{n_all:3d} → calib{len(calib):3d}"
              f"(gallery{a.k}+评审{len(review):3d}) held-out{len(evg):3d}"
              f" 缺陷{len(bad):3d}")
        for arm in ARMS + ['full']:
            v = _final(res, arm)
            tag = f" 末轮 AUROC {v.mean():5.1f}±{v.std():3.1f}"
            if arm in rates and rates[arm]:
                tag += f"  逐尺度插入率均值 {np.mean(rates[arm])*100:4.1f}%"
            print(f"    {arm:12s}{tag}")
        print()

    print("=" * 88)
    print("held-out image AUROC @ 末轮【多尺度】")
    print("=" * 88)
    print(f"{'类':13s} " + " ".join(f"{x:>12s}" for x in ARMS + ['full']))
    for cls, (res, *_n) in summary.items():
        print(f"{cls:13s} " + " ".join(
            f"{_final(res,x).mean():6.1f}±{_final(res,x).std():3.1f}"
            for x in ARMS + ['full']))
    print()
    for arm in ARMS + ['full']:
        vs = [_final(r, arm).mean() for r, *_ in summary.values()]
        print(f"  {arm:12s} {np.mean(vs):6.1f}   类数={len(vs)}  "
              f"(逐类 {', '.join(f'{x:.1f}' for x in vs)})")
    print("  ★ 'full' = 上界参照(全部评审池入库,复核预算无限),不假装可达")

    print()
    print("=" * 88)
    print("逐轮曲线(跨类均值,轮次归一化)")
    print("=" * 88)
    grid = np.linspace(0, 1, 11)
    for arm in ('none', 'gated', 'rnd_matched', 'ungated'):
        if arm == 'none':
            print(f"  {arm:12s} 恒定 "
                  f"{np.mean([np.mean(r['none']) for r,*_ in summary.values()]):5.1f}")
            continue
        acc = [np.array(c)[(grid * (len(c) - 1)).round().astype(int)]
               for r, *_ in summary.values() for c in r[arm]]
        print(f"  {arm:12s} " + " ".join(f"{x:5.1f}" for x in np.mean(acc, 0)))
    print("  轮次(归一化)  " + " ".join(f"{g:5.2f}" for g in grid))
    return 0


def _auroc(s, n_evg):
    y = np.r_[np.zeros(n_evg), np.ones(len(s) - n_evg)]
    return float(roc_auc_score(y, s) * 100)


def _final(res, arm):
    return np.array(res[arm], float) if arm in ('none', 'full') \
        else np.array([c[-1] for c in res[arm]])


def _verify(a, covs, cls):
    """同源自检:本脚本的合并分 == shot_scaling.few_map_of 再取 max。"""
    import importlib.util
    for nm in ('ss',):
        sp = importlib.util.spec_from_file_location(
            nm, str(HERE / "shot_scaling.py"))
        m = importlib.util.module_from_spec(sp)
        sys.modules[nm] = m
        sp.loader.exec_module(m)
    ss = sys.modules['ss']
    ss.IDX3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    ss.IDX2 = np.load(Path(a.deploy) / "win_idx_k2.npy")
    RS = np.random.RandomState(a.seed)
    ca = load(Path(a.calib) / cls)
    perm = RS.permutation(len(ca))
    h = len(ca) // 2
    evg = [ca[i] for i in perm[h:2 * h]]
    calib = [ca[i] for i in perm[:h]]
    sel = RS.choice(len(calib), size=a.k, replace=False)
    gal = ss.make_gal(calib, sel)
    ref = np.array([ss.stats_of(ss.few_map_of(z, gal))['max'] for z in evg])
    banks = [gal['patch'].astype(np.float32), gal['large'].astype(np.float32),
             gal['mid'].astype(np.float32)]
    qs = [np.concatenate([tokens(z, k, a.deploy) for z in evg])
          .astype(np.float32) for k, _g, _d in SCALES]
    ntok = [q.shape[0] // len(evg) for q in qs]
    bs = [np.full((len(evg), n), -np.inf, np.float32) for n in ntok]
    for bt, q, bk in zip(bs, qs, banks):
        upd(bt, q, bk)
    mine = merged_score(bs, covs)
    print(f"[verify] {cls} k={a.k} n={len(evg)} 尺图")
    print(f"  shot_scaling.few_map_of → max : 前5 {np.round(ref[:5],6)}")
    print(f"  本脚本 merged_score          : 前5 {np.round(mine[:5],6)}")
    print(f"  最大绝对误差 = {np.abs(ref-mine).max():.3e}  "
          f"全部一致 = {np.allclose(ref, mine, atol=1e-5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
