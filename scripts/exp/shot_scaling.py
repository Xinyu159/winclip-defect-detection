"""gallery 规模扫描 + 图像级打分臂对照 —— 图像级瓶颈的判定实验。

## 这一个实验要回答什么

MVTec 5/6 类上的已确立结果(scripts/exp/imgscore_arms.py 与本脚本同源):

    fewmax 随 k 陡升未饱和(MVTec 上 screw 70.5→90.7,capsule 64.4→90.9),
    而 v1(论文 Eq.6 等权平均)几乎不动(screw 77.5→81.8)。
    `cls` 与 k 无关 → **v1 里那个恒定权重的 cls 项,把加样本的收益按回一半。**

    → 路线判定:瓶颈是**记忆库/检索覆盖**,不是 CLIP 特征(路线 A),
      且 v1 的固定等权是**第二个约束**。

本脚本把同一套测法搬到任意数据集(MVTec / Surface Defects-4i)。

## ★ 过杀率的同义反复 —— 这个坑必须结构性避免

早先版本报"过杀率@P95",而阈值**就是良品分数的 P95** → "超过 P95 的良品比例"
恒等于 5%。那是**度量伪影**,不是性能。已作废过一次。

本脚本的分工(照产线实际做法):
    calib 集(commissioning)  → 出 gallery + 出阈值 τ
        ★ 且**把抽中做 gallery 的那 k 张剔除**,否则它们与 gallery 自比、
          分数被系统性压低,会把 τ 定低(过杀率虚低)
    eval-good(held-out)       → 量**真实过杀率**(未参与定阈值,不是同义反复)
    defects                   → 量漏检率 + AUROC

两个集必须**物理分开**。4i 天然满足(train/good 与 test/good 是不同目录);
MVTec 只在 test/good 里有良品,故退化为对半切(脚本自动处理并标注)。

## 硬规则

  - 阈值、校准一律只来自良品;缺陷侧只被评分,从不参与任何拟合
  - 报告里每个数字都标注 n(良品数/缺陷数),不报"凭印象的均值"
  - 良品只有 N 张时,过杀率的最小非零粒度是 1/N —— 报表必须写出 N
"""
import argparse
import sys
from pathlib import Path

# ★ 早先这里是硬编码的本地绝对路径("/home/asus/桌面/JD/...") —— 本地跑得通、
#   远端直接崩。且 Python 3 只把**脚本所在目录**放进 sys.path(不是 CWD),
#   所以 scripts/exp/ 下 import runtime 必须显式把仓库根加进来。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                     # noqa: E402
from runtime.pipeline import N_PATCH, OVPipeline as P  # noqa: E402
from sklearn.metrics import roc_auc_score              # noqa: E402

harm = P._scatter_harmonic
few = P._few_token_score

# 窗口 token 索引(1..225,不含 CLS)。与 engine_base._load_window_indices 同源
# —— 两后端共用同一份 .npy,这里直接从 deploy 目录读,不必起引擎。
# ★ 早先 few_map_of 把这份索引写成了字面量 None,`_scatter_harmonic` 里
#   `win_idx - 1` 立刻 TypeError。所以本脚本的窗口臂**从来没真正跑通过**;
#   现在改成显式载入,并由 --deploy 指定来源。
IDX3 = IDX2 = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes")
    ap.add_argument("shots", help="逗号分隔的 gallery 规模,如 1,2,4,8,16,32")
    ap.add_argument("reps", type=int, nargs="?", default=3)
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--calib", default="/tmp/feat_cache_good",
                    help="commissioning 良品(gallery 候选 + 阈值/校准来源)")
    ap.add_argument("--eval-good", default=None,
                    help="held-out 良品(只用于量过杀率与 AUROC)。"
                         "不给则把 --calib 对半切(并会在输出里标注)")
    ap.add_argument("--cache", default="/tmp/feat_cache", help="缺陷特征缓存")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--min-ref", type=int, default=20,
                    help="阈值参考集(ref = calib − gallery)的最小张数。"
                         "低于此值的 k **不做**,在表里记 '不可用'。"
                         "为什么必须有这道闸:4i 的 MT_Fray 池只有 16 张,"
                         "k=32 时 min() 截断成 16 → holdout 空 → ecdf 除零 → "
                         "静默出 NaN,看起来像'跑过了但结果是 nan'。")
    return ap.parse_args()


def load(d, n=None):
    fs = sorted(Path(d).glob("[0-9]*.npz"))
    return [np.load(f) for f in (fs[:n] if n else fs)]


def few_map_of(z, gal):
    loc = few(z['full'][1:], gal['patch']).copy()
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in (("w3", IDX3, "large"), ("w5", IDX2, "mid")):
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        loc[prf] += mf[prf]
        den[prf] += 1.0
    return loc / den


def make_gal(imgs, sel):
    return {'patch': np.concatenate([imgs[i]['full'][1:] for i in sel]),
            'large': np.concatenate([imgs[i]['w3'] for i in sel]),
            'mid': np.concatenate([imgs[i]['w5'] for i in sel])}


def stats_of(fm):
    s = np.sort(fm)[::-1]
    n = len(s)
    med = float(np.median(fm))
    return {'max': float(s[0]), 'prom': float(s[0] - med),
            'top5': float(s[:max(1, int(round(n * 0.05)))].mean())}


def ecdf(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side='right') / len(r)


ARMS = ['cls', 'fewmax', 'prom', 'top5', 'v1', 'v1raw',
        'calib_avg', 'calib_max', 'calib_noor']
# v1    = (重标定后的 cls + fewmax)/2。cls 先按 ref 的 min/ptp 拉到 [0,1] ——
#         cls_prob 是概率、fewmax 是距离,量纲不同,直接等权平均没有意义。
# v1raw = (cls_prob 原值 + fewmax)/2 —— **这才是 winclip.py 里 evaluate.py
#         --shots k 用的那一式**(`(cls_prob + few_map.max())/2`)。
#         ★ 加它就是为了让本表能和已发表的 1-shot/4-shot 数字直接对上;
#           v1 对不上是设计使然,不是 bug。


def main() -> int:
    global IDX3, IDX2
    a = parse_args()
    for k, name in ((3, "IDX3"), (2, "IDX2")):
        p = Path(a.deploy) / f"win_idx_k{k}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"窗口索引缺失: {p}\n"
                f"  → OV 侧: python scripts/export_openvino_local.py\n"
                f"  → ONNX 侧: python scripts/export_onnx_dyn.py")
        if k == 3:
            IDX3 = np.load(p)
        else:
            IDX2 = np.load(p)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    shots = [int(x) for x in a.shots.split(",")]
    RS = np.random.RandomState(42)

    print(f"gallery 规模扫描 | reps={a.reps} | k∈{shots}")
    print(f"  calib  = {a.calib}")
    print(f"  eval   = {a.eval_good or '(未给 → 由 calib 对半切,已标注)'}")
    print(f"  cache  = {a.cache}")
    print()

    summary = {}
    dropped_all = {}
    for cls in classes:
        tp = Path(a.text) / f"{cls}.npz"
        if not tp.exists():
            print(f"{cls:13s} 无文本原型 {tp},跳过")
            continue
        d = np.load(tp)
        pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])

        calib = load(Path(a.calib) / cls)
        if a.eval_good:
            evg = load(Path(a.eval_good) / cls)
            split_note = "独立"
        else:
            perm = RS.permutation(len(calib))
            h = len(calib) // 2
            evg = [calib[i] for i in perm[h:2 * h]]
            calib = [calib[i] for i in perm[:h]]
            split_note = "对半切"
        bad = [np.load(f) for f in sorted((Path(a.cache) / cls).glob("*.npz"))
               if f.name != "gallery.npz"]
        bad = [z for z in bad if z['gt'].any()]
        if not bad or len(calib) < 4 or len(evg) < 4:
            print(f"{cls:13s} 缓存不全(缺陷{len(bad)} calib{len(calib)} "
                  f"eval{len(evg)}),跳过")
            continue

        def cls_of(zs):
            return np.array([float(P._prob(z['full'][:1], pos, neg, temp)[0])
                             for z in zs])

        cp_c, cp_e, cp_b = cls_of(calib), cls_of(evg), cls_of(bad)

        # ★ 可行性闸门:ref 集(calib − gallery)不够 min_ref 张的 k 直接不做。
        #   不这样做的话 min(k, len) 会截断 gallery,把 ref 挤空 → ecdf 除零
        #   → 该格静默变 NaN,报告里看起来"跑了但没数",比明说"不可用"更坏。
        usable, dropped = [], {}
        for k in shots:
            n_ref = len(calib) - min(k, len(calib))
            if n_ref < a.min_ref:
                dropped[k] = f"池{len(calib)}张,ref仅{n_ref}<{a.min_ref}"
            else:
                usable.append(k)

        res = {arm: {k: [] for k in shots} for arm in ARMS}
        for r in range(a.reps):
            for k in usable:
                sel = RS.choice(len(calib), size=min(k, len(calib)),
                                replace=False)
                gal = make_gal(calib, sel)
                holdout = [i for i in range(len(calib)) if i not in set(sel)]

                sc = {arm: {} for arm in ARMS}
                for tag, idxs, imgs, cps in (('ref', holdout, calib, cp_c),
                                             ('evg', range(len(evg)), evg, cp_e),
                                             ('bad', range(len(bad)), bad, cp_b)):
                    st = [stats_of(few_map_of(imgs[i], gal)) for i in idxs]
                    sc[tag] = {
                        'cls': cps[list(idxs)] if tag != 'ref'
                               else cp_c[holdout],
                        'fewmax': np.array([s['max'] for s in st]),
                        'prom': np.array([s['prom'] for s in st]),
                        'top5': np.array([s['top5'] for s in st]),
                    }
                # ★ v1raw 必须在 cls 被重标定**之前**算(要的是 cls_prob 原值)
                for tag in ('evg', 'bad'):
                    sc[tag]['v1raw'] = (sc[tag]['cls'] + sc[tag]['fewmax']) / 2

                # 校准参考 = **不含 gallery 的 calib**(阈值与校准同源,符合产线)
                ref = sc['ref']
                for tag in ('evg', 'bad'):
                    qc = ecdf(ref['cls'], sc[tag]['cls'])
                    qf = ecdf(ref['fewmax'], sc[tag]['fewmax'])
                    sc[tag]['calib_avg'] = (qc + qf) / 2
                    sc[tag]['calib_max'] = np.maximum(qc, qf)
                    sc[tag]['calib_noor'] = 1 - (1 - qc) * (1 - qf)
                    sc[tag]['cls'] = (sc[tag]['cls'] - ref['cls'].min()) / \
                        (np.ptp(ref['cls']) + 1e-12)   # 仅作参照,见 docstring
                for tag in ('evg', 'bad'):
                    sc[tag]['v1'] = (sc[tag]['cls'] + sc[tag]['fewmax']) / 2

                # AUROC 的良品侧 = **只用 evg(held-out)**,不掺 ref。
                #   ref 是校准臂的拟合参考,把 ref 也算进良品侧,校准臂在自身上的
                #   q 恒为均匀分布 → 该臂被系统性地拉向 50,与外层臂不可比。
                #   evg 不参与 gallery/阈值/校准的任何一步,用它才同口径。
                # ★ 早先这里掺了 ref,且用未完成的 _ref_arm 占位返回 NaN,
                #   结果所有 calib_* 臂在 roc_auc_score 处直接抛 "Input contains NaN"
                #   —— 即**校准臂从未真正进入过 AUROC**。
                for arm in ARMS:
                    g_all = sc['evg'][arm]
                    res[arm][k].append(
                        roc_auc_score(np.r_[np.zeros(len(g_all)),
                                            np.ones(len(sc['bad'][arm]))],
                                      np.r_[g_all, sc['bad'][arm]]) * 100)
        summary[cls] = res
        dropped_all[cls] = dropped
        line = (f"{cls:13s} 缺陷{len(bad):3d} calib{len(calib):3d} "
                f"eval{len(evg):3d} ({split_note})")
        if dropped:
            line += "  ★跳过:" + ",".join(f"k={k}({r})"
                                          for k, r in dropped.items())
        print(line)

    # ---- 汇总 ----
    print()
    print("=" * 78)
    print("img AUROC(均值±标准差)")
    print("=" * 78)
    print(f"{'类':13s} {'臂':8s} " + " ".join(f"{'k='+str(k):>11s}"
                                              for k in shots))
    for cls, res in summary.items():
        for arm in ARMS:
            cells = []
            for k in shots:
                v = np.array(res[arm][k])
                cells.append(f"{v.mean():5.1f}±{v.std():3.1f}"
                             if len(v) else "—")
            print(f"{cls:13s} {arm:8s} " + " ".join(f"{c:>11s}" for c in cells))
        print()

    print("=" * 78)
    print("跨类均值(**逐 k 独立平均**:每个 k 只用在该 k 上有数的类)")
    print("=" * 78)
    print(f"{'臂':8s} " + " ".join(f"{'k='+str(k):>9s}" for k in shots)
          + f"  {'Δ(首末可用)':>14s}" + "  覆盖类数")
    for arm in ARMS:
        m, ncls = [], []
        for k in shots:
            vs = [np.mean(r[arm][k]) for r in summary.values() if r[arm][k]]
            m.append(float(np.mean(vs)) if vs else np.nan)
            ncls.append(len(vs))
        # Δ 只在**首末都可用**时才有意义;任一缺就记 '—',不拿 nan 参与算术
        ok = [i for i, x in enumerate(m) if not np.isnan(x)]
        delta = f"{m[ok[-1]] - m[ok[0]]:+14.1f}" if len(ok) >= 2 else f"{'—':>14s}"
        cells = " ".join(f"{x:9.1f}" if not np.isnan(x) else f"{'—':>9s}"
                         for x in m)
        print(f"{arm:8s} {cells}{delta}   "
              + ",".join(str(c) for c in ncls))
    print("  ★ 覆盖类数逐 k 递减是**数据本身的约束**(小池类达不到大 k),"
          "不是丢数。")
    print("  ★ 跨类均值因此在不同 k 上不是同一批类 —— 读趋势只看单类表,"
          "跨类均值仅供粗看。")

    if any(dropped_all.values()):
        print("=" * 78)
        print(f"★ 被闸门跳过的格子(gallery 池不够 min_ref={a.min_ref} 张留给阈值参考)")
        print("=" * 78)
        for cls, dd in dropped_all.items():
            if dd:
                print(f"  {cls:13s} " + " | ".join(f"k={k}: {r}"
                                                   for k, r in dd.items()))
        print("  → 报表里这些格子写'不可用'并注明原因,**不是** 0,也不是 nan。")
        print()

    print()
    print("判据:")
    print("  fewmax 随 k 陡升未饱和 → 瓶颈是记忆库(路线 A):加良品样本有效")
    print("  v1 的 Δ 远小于 fewmax 的 Δ → 固定等权把加样本的收益按回去了")
    print("  cls 恒定(与 k 无关)→ 它是那个'不动'的拖累项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
