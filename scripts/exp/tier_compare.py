"""决定性对比:多尺度窗口塔到底值多少?

与真实 pipeline 的逐位对拍见 verify_rebuild.py(max|Δ|=0)。

**本脚本报两套数字,不可混用:**

  [定位口径] 只用 few 分支 —— 每个 patch 的分数只由"该 patch 邻域被精检到的
             局部证据"决定。这才是"缺陷在图上哪个位置"的度量,也是产线做
             复检/追溯/标注时真正需要的口径。
             高分完全来自 few_map,不掺任何全局标量。

  [判定口径] 真实 pipeline 的 m_all = mix_zero(cls_prob, 窗口) + mix_few(...)。
             mix_zero 在无窗口时**整幅等于 cls_prob**(一个全局标量),
             缺陷图恒高、良品图恒低 —— 它是"这张图有问题吗"的度量,
             不是"哪里有问题"的度量。

  ★ L1 档在 [定位口径] 下**并非**常数:few 分支的地基是 patch 尺度的 few 分
    (`few(full[1:], gal['patch'])`),它逐 patch 不同,单尺度就有 94 左右。
    (早期版本这里写着"L1 恒为常数、AUROC=50",那是把 m_zero 的常数性
     错安到了 few 分支上 —— 代码从来没那样算过,是注释在说谎。)

三个档位(token 预算):
  L1       仅整图(226 token)
  L1+3x3   + 3×3 全量窗口(1916 token)
  full     + 3×3 + 2×2 全量(2896 token)—— research 路径
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

TOK = {'L1': 226, 'L1+3x3': 1916, 'full': 2896}
LEVELS = ('L1', 'L1+3x3', 'full')


def build(z, gal, pos, neg, temp, level):
    full = z['full']
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])
    patch_score = few(full[1:], gal['patch'])
    few_map = patch_score.copy()          # den=1 的基底
    inv = np.full(N_PATCH, 1.0 / max(cls_prob, 1e-12), np.float32)
    nt = np.ones(N_PATCH, np.float32)
    # 关键:**L1 不加任何窗口尺度**。
    # 这里曾写成 scales 无条件从 [w3] 开始,导致 "L1" 与 "L1+3×3" 实际
    # 计算的是同一个东西,于是得出"两者逐位完全相同"的**假结论**。
    # 一个档位定义错了,整张对比表就失去意义 —— 档位必须严格按 token 预算划分。
    scales = []
    if level in ('L1+3x3', 'full'):
        scales.append(('w3', idx3, 'large'))
    if level == 'full':
        scales.append(('w5', idx2, 'mid'))
    for key, idx, gname in scales:
        wp = P._prob(z[key], pos, neg, temp)
        m, c = harm(wp, idx)
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        nt[pr] += 1.0
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        few_map[prf] += mf[prf]
    # few_map 的分母:每个尺度覆盖到的 patch +1
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in scales:
        _, cf = harm(np.zeros(len(idx), np.float32), idx)
        den[cf > 0] += 1.0
    few_map = few_map / den
    m_zero = nt / inv
    m_all = m_zero + few_map          # 判定口径(含全局标量 cls_prob)
    img_score = (cls_prob + float(few_map.max())) / 2.0
    return m_all.reshape(GRID, GRID), img_score, few_map.reshape(GRID, GRID)


# 过杀率的分母是良品张数,而良品只有 20~41 张 —— P99 阈值实际落在"最大值附近",
# 超过它的恒为 1 张,于是过杀率被量化为 1/N(20 张 → 5.0%,28 张 → 3.6%)。
# 那是**样本量的伪影,不是过杀水平**。故 P95 与 P99 并报,并注明 N。
print(f"{'类':10s} {'档位':8s} {'token':>6s} {'定位pix':>8s} {'判定pix':>8s} "
      f"{'img':>6s} {'过杀P95':>8s} {'漏检P95':>8s} {'过杀P99':>8s} {'漏检P99':>8s}")
summary = {k: {} for k in LEVELS}
raw = {}
for cls in ['tile', 'carpet', 'bottle', 'metal_nut', 'screw']:
    d = np.load(f'data/deploy/text_protos/{cls}.npz')
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    bad = [np.load(f) for f in sorted((CD / cls).glob('*.npz'))
           if f.name != 'gallery.npz']
    bad = [z for z in bad if z['gt'].any()]
    good = [np.load(f) for f in sorted((CG / cls).glob('*.npz'))]
    print(f"# {cls}: 缺陷 {len(bad)} 张,良品 {len(good)} 张")
    for lv in LEVELS:
        ps, pl, pg, gs, bs = [], [], [], [], []
        for z in bad:
            m, s, lm = build(z, gal, pos, neg, temp, lv)
            pg.append(z['gt'].flatten())
            ps.append(upsample_bilinear_np(m, 240, 240).flatten())
            pl.append(upsample_bilinear_np(lm, 240, 240).flatten())
            bs.append(s)
        for z in good:
            gs.append(build(z, gal, pos, neg, temp, lv)[1])
        gs, bs = np.array(gs), np.array(bs)
        pgc = np.concatenate(pg)
        loc = roc_auc_score(pgc, np.concatenate(pl)) * 100
        det = roc_auc_score(pgc, np.concatenate(ps)) * 100
        img = roc_auc_score(np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                            np.r_[gs, bs]) * 100
        cols = []
        for p in (95, 99):
            thr = float(np.percentile(gs, p))
            cols += [(gs > thr).mean() * 100, (bs <= thr).mean() * 100]
        summary[lv][cls] = dict(loc=loc, det=det, img=img,
                               n_good=len(gs), n_bad=len(bs),
                               over95=cols[0], esc95=cols[1],
                               over99=cols[2], esc99=cols[3])
        raw[f"{cls}|{lv}"] = dict(good=gs.tolist(), bad=bs.tolist())
        print(f"{cls:10s} {lv:8s} {TOK[lv]:6d} {loc:8.1f} {det:8.1f} {img:6.1f} "
              f"{cols[0]:7.1f}% {cols[1]:7.1f}% {cols[2]:7.1f}% {cols[3]:7.1f}%")
    print()

Path('results/tier_scores.json').write_text(
    __import__('json').dumps(raw, indent=1))

print("=== 5 类均值 ===")
print(f"{'档位':8s} {'token':>6s} {'相对L1':>7s} {'定位pix':>8s} {'判定pix':>8s} "
      f"{'img':>6s} {'过杀P95':>8s} {'漏检P95':>8s}")
for lv in LEVELS:
    f = lambda k: np.mean([summary[lv][c][k] for c in summary[lv]])
    print(f"{lv:8s} {TOK[lv]:6d} {TOK[lv]/226:6.1f}x {f('loc'):8.1f} "
          f"{f('det'):8.1f} {f('img'):6.1f} {f('over95'):7.1f}% {f('esc95'):7.1f}%")
