"""决定性验伪:我的重建,到底等不等于真实 research 路径?

前一版重建用的是"patch few 分当地基",与真实 pipeline 差 68~81%(已证伪)。
对照 runtime/pipeline.py 的真值公式:

    # zero 分支(地基是 **cls_prob**,不是 patch 分!)
    inv = full(N_PATCH, 1/cls_prob); n_terms = ones(N_PATCH)
    for 每个尺度(3x3, 2x2):
        m, cnt = harm(w_prob, idx)
        inv[cnt>0] += 1/m[cnt>0]; n_terms[cnt>0] += 1
    m_zero = n_terms / inv

    # few 分支(与 zero **相加**,不是替代)
    few_num = few(full[1:], gal['patch']).copy(); few_den = ones(N_PATCH)
    for 每个尺度:
        m, cnt = harm(few(ws, g[scale]), idx)
        few_num[cnt>0] += m[cnt>0]; few_den[cnt>0] += 1
    few_map = few_num / few_den

    m_all     = m_zero + few_map
    img_score = (cls_prob + few_map.max()) / 2
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from PIL import Image
from runtime.ov_engine import OVEngine
from runtime.pipeline import GRID, N_PATCH, OVPipeline
import mvtec

idx3 = np.load('data/deploy/win_idx_k3.npy')
idx2 = np.load('data/deploy/win_idx_k2.npy')
harm = OVPipeline._scatter_harmonic
few = OVPipeline._few_token_score

DE = 'data/deploy'
ROOT = 'data/mvtec_anomaly_detection'
CLS = 'tile'
CD = Path('/tmp/feat_cache')

eng = OVEngine(DE, device='CPU')
pipe = OVPipeline(eng, "data/deploy/text_protos")
pipe.set_class(CLS)
gal = dict(np.load(CD / CLS / 'gallery.npz'))
pipe.gallery = gal


def mix_zero(cls_prob, terms):
    """zero 分支:地基是 cls_prob,窗口分并入。"""
    inv = np.full(N_PATCH, 1.0 / max(cls_prob, 1e-12), np.float32)
    nt = np.ones(N_PATCH, np.float32)
    for m, c in terms:
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        nt[pr] += 1.0
    return nt / inv


def mix_few(patch_score, terms):
    """few 分支:地基是 patch 尺度的 few 分。"""
    num = patch_score.copy()
    den = np.ones(N_PATCH, np.float32)
    for m, c in terms:
        pr = c > 0
        num[pr] += m[pr]
        den[pr] += 1.0
    return num / den


def rebuild_full(z):
    """同 rebuild,但把 few_map 单独交出来(定位口径用的就是它)。"""
    full = z['full']
    cls_prob = float(pipe._prob(full[:1], pipe.pos, pipe.neg, pipe.temp)[0])
    zt = [harm(pipe._prob(z['w3'], pipe.pos, pipe.neg, pipe.temp), idx3),
          harm(pipe._prob(z['w5'], pipe.pos, pipe.neg, pipe.temp), idx2)]
    ft = [harm(few(z['w3'], gal['large']), idx3),
          harm(few(z['w5'], gal['mid']), idx2)]
    few_map = mix_few(few(full[1:], gal['patch']), ft)
    return mix_zero(cls_prob, zt), few_map, cls_prob


def main():
    print("=== 第一节:m_all / img_score 逐位对拍 ===")
    print(f"{'图':>4s} {'真实.max':>10s} {'重建.max':>10s} {'map max|D|':>12s} "
          f"{'真实img_score':>13s} {'重建img_score':>13s} {'score D':>10s}")
    for i, (name, rel, ip, mp) in enumerate(mvtec.iter_test_images(ROOT, CLS)):
        if mp is None or i >= 5:
            continue
        img = np.asarray(Image.open(ip).convert('RGB'))
        real, real_s, _ = pipe.anomaly_maps(eng.preprocess_rgb(img), use_few=True)
        m_zero, few_map, cls_prob = rebuild_full(np.load(CD / CLS / f"{i:03d}.npz"))
        mine = (m_zero + few_map).reshape(GRID, GRID)
        mine_s = (cls_prob + float(few_map.max())) / 2.0
        d = float(np.abs(real - mine).max())
        print(f"{i:4d} {real.max():10.6f} {mine.max():10.6f} {d:12.2e} "
              f"{real_s:13.6f} {mine_s:13.6f} {abs(real_s-mine_s):10.2e}")
    print("判据:max|D| ~1e-7 -> 重建正确;>1e-2 -> 仍错")

    # ------------------------------------------------------------------
    # 第二节:定位口径 few_map 是不是真的定位分数
    #
    # pipeline 把 few_map 算成局部变量,但它可从两次公开调用**精确还原**:
    #     m_all_few  = m_zero + few_map
    #     m_all_zero = m_zero
    #  →  few_map = m_all_few - m_all_zero   (逐位,不依赖我的任何重建)
    #
    # 拿到真值后回答报告 §4-3:把它当定位分数算像素 AUROC,是否站得住。
    # ------------------------------------------------------------------
    print("\n=== 第二节:定位口径 few_map 的正确性(与真值对拍) ===")
    print(f"{'图':>4s} {'还原.max':>10s} {'重建.max':>10s} {'max|D|':>12s} "
          f"{'零few差':>9s}")
    n2 = 0
    for i, (name, rel, ip, mp) in enumerate(mvtec.iter_test_images(ROOT, CLS)):
        if mp is None or n2 >= 5:
            continue
        n2 += 1
        x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert('RGB')))
        real_few, _, _ = pipe.anomaly_maps(x, use_few=True)
        real_zero, _, _ = pipe.anomaly_maps(x, use_few=False)
        real_fewmap = real_few - real_zero                      # 真值
        _, mine_fewmap, _ = rebuild_full(np.load(CD / CLS / f"{i:03d}.npz"))
        mine_fewmap = mine_fewmap.reshape(GRID, GRID)
        gap = float(np.abs(real_zero).max())                    # m_zero 的量级
        print(f"{i:4d} {real_fewmap.max():10.6f} {mine_fewmap.max():10.6f} "
              f"{float(np.abs(real_fewmap-mine_fewmap).max()):12.2e} {gap:9.4f}")

    print("\n判据:max|D| ~1e-7 -> 我的 mix_few 与 pipeline 逐位一致,定位口径可信")
    print("     max|D| >1e-2 -> 定位口径的数字不能用,报告 §4-3 需改写")
    print("\n注:'零few差'是该图 m_zero 的最大值,即全局标量 cls_prob 抬高的量级。")
    print("    把 m_all 当定位分数时,误差正是这个量级 —— 这正是 v2 假结论的来源。")


if __name__ == "__main__":
    main()
