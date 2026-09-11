"""测"窗口的 zero-shot 文本概率"到底有没有用 —— metal_nut 反向的真正嫌疑。

背景链:
  1. 早先已证:`_prob(full[1:], 文本原型)` 与 GT **反相关**(5 类 14.3/7.3/12.7/19.3/16.2)
  2. 但窗口路径一直在用 `_prob(z['w3'], pos, neg, temp)` —— 同一种"特征 vs 文本原型"
     的打分,只是特征来自窗口而不是整图 mosaic 令牌
  3. L1 档 m_zero 恒等于 cls_prob(纯常数),加窗口后 m_zero 变成逐 patch 变化
  4. 而 few 分支(定位)是**改善**的(metal_nut 90.2→92.2)
  → 故 metal_nut 判定口径 −4.2 只可能来自 m_zero 被窗口 zero 概率污染

本脚本单独给"窗口 zero 概率"打分:
  若 pixel AUROC < 50,说明它**反相关**,是纯粹的噪声/有害项,
  那么把它并进 m_zero 就是负贡献 —— metal_nut 的反向就有了机制解释。

对照 arm:
  zero_win  = 只用窗口 zero 概率(即 m_zero 去掉 cls_prob 地基)
  few_win   = 只用窗口 few 分数(定位口径)
  cls       = 只用整图 cls_prob(常数,pooled 时等于"图像级"信号)
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

print(f"{'类':11s} {'档位':8s} {'zero_win':>9s} {'few_win':>9s} {'cls(常数)':>10s}")
for cls in CLASSES:
    if not Path(f'data/deploy/text_protos/{cls}.npz').exists():
        continue
    if not (CD / cls / 'gallery.npz').exists():
        print(f"{cls:11s} 缓存不全,跳过")
        continue
    d = np.load(f'data/deploy/text_protos/{cls}.npz')
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    bad = [np.load(f) for f in sorted((CD / cls).glob('*.npz'))
           if f.name != 'gallery.npz']
    bad = [z for z in bad if z['gt'].any()]
    good = [np.load(f) for f in sorted((CG / cls).glob('*.npz'))]
    if not bad or not good:
        print(f"{cls:11s} 缓存不全(缺陷{len(bad)} 良品{len(good)}),跳过")
        continue

    for lv in ('L1+3x3', 'full'):
        zs, fs, pg, cs, cov = [], [], [], [], []
        for z in bad + good:
            full = z['full']
            scales = [('w3', idx3, 'large')]
            if lv == 'full':
                scales.append(('w5', idx2, 'mid'))
            # zero_win:窗口文本概率,与 m_zero 里并入的是同一个量
            zw_num = np.zeros(N_PATCH, np.float32)
            zw_den = np.zeros(N_PATCH, np.float32)
            fw_num = np.zeros(N_PATCH, np.float32)
            for key, idx, gname in scales:
                wp = P._prob(z[key], pos, neg, temp)
                m, c = harm(wp, idx)
                pr = c > 0
                zw_num[pr] += m[pr]
                zw_den[pr] += 1.0
                mf, cf = harm(few(z[key], gal[gname]), idx)
                prf = cf > 0
                fw_num[prf] += mf[prf]
            zw = np.where(zw_den > 0, zw_num / np.maximum(zw_den, 1e-12), np.nan)
            fw = np.where(zw_den > 0, fw_num / np.maximum(zw_den, 1e-12), np.nan)
            cs.append(float(P._prob(full[:1], pos, neg, temp)[0]))
            # 先上采样到 240×240 再打掩膜 —— GT 就是这个尺度(57600 展平),
            # 拿 15×15 的 mask 去索引 57600 会报 shape 错(踩过一次)。
            # 上采样会把未覆盖 patch 的 nan 扩散到邻域,故只在**完全覆盖**处比较。
            zs.append(zw)
            fs.append(fw)
            cov.append(zw_den.reshape(GRID, GRID) > 0)
            if 'gt' in z.files:
                pg.append(z['gt'])
            else:
                pg.append(None)
        nb = len(bad)
        # 3×3 窗口布局覆盖全图,故 cov 实际全 True;仍显式用 mask 以保证
        # 若将来改成稀疏选窗,这里不会静默地把未覆盖区域算进去
        covs = np.stack(cov)
        # covm 是 15×15,GT 是 240×240 —— 上采样 covm 到同一尺度再取
        covu = [upsample_bilinear_np(covm.astype(np.float32), 240, 240) > 0.5
                for covm in covs[:nb]]
        pgc = np.concatenate([p[cu] for p, cu in zip(pg[:nb], covu)])
        zu = np.concatenate([upsample_bilinear_np(z.reshape(GRID, GRID), 240, 240)[cu]
                             for z, cu in zip(zs[:nb], covu)])
        fu = np.concatenate([upsample_bilinear_np(f.reshape(GRID, GRID), 240, 240)[cu]
                             for f, cu in zip(fs[:nb], covu)])
        za = roc_auc_score(pgc, zu) * 100
        fa = roc_auc_score(pgc, fu) * 100
        # cls 是每图常数 → 对 pooled 像素 AUROC 等于图像级 AUROC
        cs = np.array(cs)
        ca = roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(bad))],
                           np.r_[cs[len(bad):], cs[:len(bad)]]) * 100
        print(f"{cls:11s} {lv:8s} {za:9.1f} {fa:9.1f} {ca:10.1f}")

print("\n判据:zero_win < 50 → 窗口文本概率与 GT 反相关,并入 m_zero 是**负贡献**")
print("     zero_win ≈ 50 → 无信息;> 50 → 有正贡献(与早先整图结论不同)")
