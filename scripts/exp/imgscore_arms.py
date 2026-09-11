"""图像级打分 `img_score` 的拆解与升级 —— 全部离线(缓存特征),不碰 GPU。

## 为什么查这里

级联三档报告(15 类全量)得出的定论是:
    screw 定位极好(判定 pix 96.2,与 research 96.0 一致)但 img 只有 75.1、
    漏检 68~72%,**三档都一样** —— 加窗口救不了。
瓶颈在 `img_score = (cls_prob + few_map.max())/2` 这个组装方式上,
不在算力。修一个打分函数(0 额外算力)的收益 > 算力翻 12.8 倍。

这与像素级的 metal_nut 问题是**同一类错误**在两个层级上的复现:
    像素级:`m_zero + few_map` 相加 → 把随机的 zero 证据 (50.9) 掺进
            本来很好的 few 定位 (92.2) → 69.9
    图像级:`(cls_prob + max(few_map))/2` 等权平均 → 把弱的 cls 拉平强的 few

## 论文对此的说法(逐字)

Eq(6) `ascore_W(x) := ½(ascore_0(f(x)) + max_ij M^W_ij)` —— 这就是 v1,**等价权平均**。
论文 Table 5(AC,MVTec)说两项各有贡献、合起来更好(91.8 / 87.9 → 93.1),
但那是在 MVTec 全体上;等权重只在"两路信号同样可靠"时才是最优的,
而可靠性是**逐类**变化的 —— 这正是升级的入口。

## 本脚本的对照臂(全部只用良品标定,不碰 GT)

单分量(无需校准,单调变换不改变 AUROC):
    A0 cls          = cls_prob                        (论文第一项)
    A1 fewmax       = max(few_map)                    (论文第二项)
    A2 v1           = (cls_prob + fewmax)/2           (论文 Eq(6),现行)
    A3 fewtop5      = few_map 前 5% 均值              (把 max 换成稳健统计量)
    A4 fewtop25 / A5 fewmean                          (看 k 的走势)

融合臂(需校准:两路量纲不同,等权平均隐含假设了同分布):
    B0 calib_avg    = (q_cls + q_fewmax)/2            ★ 分离"没校准" vs "等权平均"
    B1 calib_max    = max(q_cls, q_fewmax)            (取更强的证据)
    B2 calib_noor   = 1−(1−q_cls)(1−q_fewmax)         (噪声或)
    B3 calib_t5_max = max(q_cls, q_fewtop5)
    B4 calib_t5_noor= 1−(1−q_cls)(1−q_fewtop5)

**B0 是关键的对照组**:它和 A2 结构完全相同,只多了分位数校准。
  B0 ≈ A2  → 校准不是问题,等权平均本身是问题 → 该用 B1/B2
  B0 ≫ A2  → 缺的是校准,不是平均

## 校准协议:两折交叉拟合(cross-fitting)

q = CDF_良品(score)。若在**用于评估的同一批良品**上估 CDF,评估会偏乐观。
故把良品分两折:标定用一折、打分用另一折,互换后合并 —— 近似无偏。
(硬规则:良品来自 test/good,与 4-shot gallery 的 train/good 天然不相交。)
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P
from sklearn.metrics import roc_auc_score

idx3 = np.load('data/deploy/win_idx_k3.npy')
idx2 = np.load('data/deploy/win_idx_k2.npy')
harm = P._scatter_harmonic
few = P._few_token_score
CD = Path('/tmp/feat_cache')
CG = Path('/tmp/feat_cache_good')

CLASSES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    'tile', 'carpet', 'bottle', 'metal_nut', 'screw']


def parts(z, gal, pos, neg, temp):
    """返回 (few_map, cls_prob) —— few_map 是**定位口径**(纯 few,不含 m_zero)。

    与 tier_compare.py 的重建逐位对拍过(m_all(use_few=True)−m_all(use_few=False),
    max|Δ| = 7.45e-09)。这里直接构造,省掉两次全量计算。
    """
    full = z['full']
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])
    loc = few(full[1:], gal['patch']).copy()
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in (('w3', idx3, 'large'), ('w5', idx2, 'mid')):
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        loc[prf] += mf[prf]
        den[prf] += 1.0
    return loc / den, cls_prob


def aggs(fm):
    """few_map(225,) → 几个图像级聚合量。"""
    s = np.sort(fm)[::-1]
    n = len(s)
    return {
        'max': float(s[0]),
        'top5': float(s[:max(1, int(round(n * 0.05)))].mean()),
        'top25': float(s[:max(1, int(round(n * 0.25)))].mean()),
        'mean': float(s.mean()),
    }


def ecdf(ref, x):
    """q = 在良品参考分布 ref 中的经验分位(0..1)。ref 排序后二分查找。"""
    r = np.sort(ref)
    return np.searchsorted(r, x, side='right') / len(r)


def auc(gs, bs):
    return roc_auc_score(np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                         np.r_[gs, bs]) * 100


def miss_at_fpr(gs, bs, fpr):
    """过杀率固定为 fpr 时的漏检率。阈值取良品分位(硬规则:来自良品)。"""
    thr = np.percentile(gs, 100 - fpr)
    return float((bs <= thr).mean() * 100)


ARMS = ['cls', 'fewmax', 'v1', 'fewtop5', 'fewtop25', 'fewmean',
        'calib_avg', 'calib_max', 'calib_noor',
        'calib_t5_max', 'calib_t5_noor']

print(f"{'类':11s} " + " ".join(f"{a:>12s}" for a in ARMS))
print(f"{'':11s} " + " ".join(f"{'img AUROC':>12s}" for _ in ARMS))
print("-" * (12 + 13 * len(ARMS)))

acc = {a: [] for a in ARMS}
acc_miss = {a: {5: [], 1: []} for a in ARMS}
rows = []
for cls in CLASSES:
    proto = Path(f'data/deploy/text_protos/{cls}.npz')
    if not proto.exists() or not (CD / cls / 'gallery.npz').exists():
        print(f"{cls:11s} 缺数据,跳过")
        continue
    d = np.load(proto)
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    bad = [np.load(f) for f in sorted((CD / cls).glob('*.npz'))
           if f.name != 'gallery.npz']
    bad = [z for z in bad if z['gt'].any()]
    good = [np.load(f) for f in sorted((CG / cls).glob('*.npz'))]
    if not bad or len(good) < 4:
        print(f"{cls:11s} 缓存不全(缺陷{len(bad)} 良品{len(good)}),跳过")
        continue

    def collect(zs):
        cp, ag = [], []
        for z in zs:
            fm, c = parts(z, gal, pos, neg, temp)
            cp.append(c)
            ag.append(aggs(fm))
        return np.array(cp), {k: np.array([a[k] for a in ag])
                              for k in ag[0]}

    cp_b, ag_b = collect(bad)
    cp_g, ag_g = collect(good)

    # ---- 两折交叉拟合:标定用一折良品,打分用另一折 ----
    ng = len(good)
    fold = np.arange(ng) % 2
    q_b = {k: np.zeros(len(bad)) for k in ('cls', 'max', 'top5')}
    q_g = {k: np.zeros(ng) for k in ('cls', 'max', 'top5')}
    for f in (0, 1):
        ref_cp = cp_g[fold != f]
        ref_ag = {k: ag_g[k][fold != f] for k in ('max', 'top5')}
        # 缺陷侧与良品侧各按自己的下标对半分(长度不同,不能用同一个 mask)
        m_b = np.arange(len(bad)) % 2 == f
        m_g = fold == f
        q_b['cls'][m_b] = ecdf(ref_cp, cp_b[m_b])
        q_b['max'][m_b] = ecdf(ref_ag['max'], ag_b['max'][m_b])
        q_b['top5'][m_b] = ecdf(ref_ag['top5'], ag_b['top5'][m_b])
        q_g['cls'][m_g] = ecdf(ref_cp, cp_g[m_g])
        q_g['max'][m_g] = ecdf(ref_ag['max'], ag_g['max'][m_g])
        q_g['top5'][m_g] = ecdf(ref_ag['top5'], ag_g['top5'][m_g])

    vals_g = {
        'cls': cp_g, 'fewmax': ag_g['max'], 'fewtop5': ag_g['top5'],
        'fewtop25': ag_g['top25'], 'fewmean': ag_g['mean'],
        'v1': (cp_g + ag_g['max']) / 2,
        'calib_avg': (q_g['cls'] + q_g['max']) / 2,
        'calib_max': np.maximum(q_g['cls'], q_g['max']),
        'calib_noor': 1 - (1 - q_g['cls']) * (1 - q_g['max']),
        'calib_t5_max': np.maximum(q_g['cls'], q_g['top5']),
        'calib_t5_noor': 1 - (1 - q_g['cls']) * (1 - q_g['top5']),
    }
    vals_b = {
        'cls': cp_b, 'fewmax': ag_b['max'], 'fewtop5': ag_b['top5'],
        'fewtop25': ag_b['top25'], 'fewmean': ag_b['mean'],
        'v1': (cp_b + ag_b['max']) / 2,
        'calib_avg': (q_b['cls'] + q_b['max']) / 2,
        'calib_max': np.maximum(q_b['cls'], q_b['max']),
        'calib_noor': 1 - (1 - q_b['cls']) * (1 - q_b['max']),
        'calib_t5_max': np.maximum(q_b['cls'], q_b['top5']),
        'calib_t5_noor': 1 - (1 - q_b['cls']) * (1 - q_b['top5']),
    }

    line = []
    for a in ARMS:
        gs, bs = vals_g[a], vals_b[a]
        acc[a].append(auc(gs, bs))
        for fpr in (5, 1):
            acc_miss[a][fpr].append(miss_at_fpr(gs, bs, fpr))
        line.append(f"{auc(gs, bs):12.1f}")
    rows.append((cls, len(bad), len(good)))
    print(f"{cls:11s} " + " ".join(line))

print("-" * (12 + 13 * len(ARMS)))
print(f"{'均值':11s} " + " ".join(f"{np.mean(acc[a]):12.1f}" for a in ARMS))
print(f"{'漏检@过杀5%':8s} " + " ".join(f"{np.mean(acc_miss[a][5]):11.1f}%"
                                        for a in ARMS))
print(f"{'漏检@过杀1%':8s} " + " ".join(f"{np.mean(acc_miss[a][1]):11.1f}%"
                                        for a in ARMS))

print()
print("=" * 78)
print("逐类看(AUROC):哪些类被等权平均拖了后腿,升级臂能捞回多少")
print("=" * 78)
hdr = ['cls', 'fewmax', 'v1', 'calib_avg', 'calib_max', 'calib_noor',
       'calib_t5_max', 'calib_t5_noor']
print(f"{'类':11s} " + " ".join(f"{a:>13s}" for a in hdr))
print(f"{'':11s} " + " ".join(f"{'':>13s}" for _ in hdr))
for cls, nb, ng in rows:
    i = [c for c, _, _ in rows].index(cls)
    print(f"{cls:11s} " + " ".join(f"{acc[a][i]:13.1f}" for a in hdr))

print()
print("判据:")
print("  B0(calib_avg) ≈ A2(v1) 且 B1/B2 ≫ A2 → **等权平均本身**是瓶颈,该换融合")
print("  B0 ≫ A2                              → 缺的是**校准**,不是平均")
print("  A1(fewmax) ≫ A2(v1)                  → cls 项是纯稀释(与像素级 metal_nut 同型)")
print("  A3(fewtop5) 与 A1 的差                → max 这个统计量本身有多脆")
