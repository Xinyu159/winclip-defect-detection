"""测 zero/few 融合方式 —— 论文没给公式,所以这里有自由度。

起因:论文原文对两个分数图的融合只有一句 "then fusing with our language-guided
prediction M̄^0W",**没有任何公式**。我们的实现选了相加:
    m_all = m_zero + few_map
而消融显示 metal_nut 的纯 few 定位是 92.2,相加后掉到 69.9(−22.3)。

论文引言解释了为什么偏偏是 metal_nut:
    "Metal-nut has an anomaly type labeled as 'flipped upside-down',
     which can only be identified relatively from a normal image."
即该类的文本原型在原理上描述不了缺陷 → zero 分支接近随机(research 日志 50.9)
→ 相加等于把随机证据均匀掺进本来很好的 few 定位。

本脚本对照四种融合(全部只用缓存特征,不碰 GPU):
    sum      现行:m_zero + few_map
    max      max(m_zero, few_map)  —— 取更强的证据
    gate     zero 置信时才并入:cls_prob 高于良品分位才用 m_zero,否则只用 few
    fewonly  完全不并入(下界参照,产线不可用:它给不出图像级分数)

★ 注意:gate 的阈值只能从**良品**标定(硬规则),不能看 GT。
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P
from scripts.eval_ov import upsample_bilinear_np
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
    """返回 (m_zero, few_map, cls_prob) —— 与 pipeline 逐位一致(已对拍)。"""
    full = z['full']
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])
    patch_score = few(full[1:], gal['patch'])
    loc = patch_score.copy()
    inv = np.full(N_PATCH, 1.0 / max(cls_prob, 1e-12), np.float32)
    nt = np.ones(N_PATCH, np.float32)
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in (('w3', idx3, 'large'), ('w5', idx2, 'mid')):
        m, c = harm(P._prob(z[key], pos, neg, temp), idx)
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        nt[pr] += 1.0
        den[pr] += 1.0
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        loc[prf] += mf[prf]
    return (nt / inv), (loc / den), cls_prob


print(f"{'类':11s} {'策略':9s} {'判定pix':>8s} {'img':>7s} {'漏检@P95':>9s}")
mean = {}
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

    # gate 的阈值:只用良品标定 —— 取良品 cls_prob 的 P50
    g_cls = np.array([float(P._prob(np.load(f)['full'][:1], pos, neg, temp)[0])
                      for f in sorted((CG / cls).glob('*.npz'))])
    gate_thr = float(np.percentile(g_cls, 50))

    for strat in ('sum', 'max', 'gate', 'fewonly'):
        pg, ps, gs, bs = [], [], [], []
        for z in bad + good:
            mz, fm, cp = parts(z, gal, pos, neg, temp)
            if strat == 'sum':
                m = mz + fm
            elif strat == 'max':
                m = np.maximum(mz, fm)
            elif strat == 'gate':
                m = (mz + fm) if cp > gate_thr else fm
            else:
                m = fm
            # 图像级:论文 Eq(6) 的形式恒不变(否则不是同一个 pipeline 了)
            s = (cp + float(fm.max())) / 2.0
            isbad = 'gt' in z.files
            if isbad:
                pg.append(z['gt'].flatten())
                ps.append(upsample_bilinear_np(m.reshape(GRID, GRID), 240, 240).flatten())
                bs.append(s)
            else:
                gs.append(s)
        gs, bs = np.array(gs), np.array(bs)
        pix = roc_auc_score(np.concatenate(pg), np.concatenate(ps)) * 100
        img = roc_auc_score(np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                            np.r_[gs, bs]) * 100
        thr = float(np.percentile(gs, 95))
        esc = (bs <= thr).mean() * 100
        mean.setdefault(strat, []).append((pix, img, esc))
        print(f"{cls:11s} {strat:9s} {pix:8.1f} {img:7.1f} {esc:8.1f}%")
    print()

print("=" * 50)
print(f"{'策略':9s} {'判定pix':>8s} {'img':>7s} {'漏检@P95':>9s}")
for strat in ('sum', 'max', 'gate', 'fewonly'):
    if strat not in mean:
        continue
    a = np.array(mean[strat])
    print(f"{strat:9s} {a[:,0].mean():8.1f} {a[:,1].mean():7.1f} {a[:,2].mean():8.1f}%")
print("\n判据:某一策略的判定pix 显著高于 sum,且 img 不掉 → 融合方式就是瓶颈")
print("    若差距主要在 metal_nut 一类 → 与论文引言'文本描述不了该缺陷'一致")
print("\n注意:gate 用的是良品 cls_prob 的 P50 作阈值,未用 GT(硬规则)")
