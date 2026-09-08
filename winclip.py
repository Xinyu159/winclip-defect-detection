"""WinCLIP(CVPR'23)完整复现核心:zero-/few-shot 异常分类与像素分割。

设计对齐论文与官方复现(mala-lab),骨干为**全冻结**的 CLIP ViT-B-16-plus-240
(LAION-400M,240px → 15×15 patch),零训练、零反向传播:

  1. 手工前向(已验证与 model.encode_image 输出余弦相似度 = 1.0):
     conv1 patch 化 → 拼 [CLS] + 位置编码 → ln_pre → 12 层 transformer →
     ln_post → proj(896→640 共享空间,与文本同空间);
  2. 文本侧 CPE(prompts.py):7 正常 + 4 异常状态词 × 22 句式模板,
     逐标签平均成 normal/abnormal 原型各 1 个向量;
  3. 多尺度窗口:2×2(32px,196 窗)/ 3×3(48px,169 窗)patch 子序列
     [CLS+窗口] 打包成 batch 一次过整塔(权重共享,窗口级 CLS 描述局部);
  4. anomaly map = 窗口"异常文本"概率按覆盖关系做调和平均 → 15×15,
     再与整图 CLS 异常概率三路调和(3/(1/m48 + 1/m32 + 1/z0));
  5. few-shot(全冻结):k 张正常参考图的三尺度特征作 gallery,查询窗口/
     patch 特征最近邻 0.5·(1−max cos) 得视觉异常;三尺度平均 → few map,
     最终 map = zero map + few map;image 分 = (文本概率 + max(few map))/2。

消融记录(详情见 memory):温度{100,14.3,1}、窗口分支数、896 无 proj 空间、
窗口内 ln_pre 等变体均无增益,最终采用上述官方默认配置。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import open_clip


class WinCLIP:
    def __init__(self, model_name: str = "ViT-B-16-plus-240",
                 weights: str = "", device: str = "cuda"):
        """weights: 本地 ckpt 路径(plus-240 权重),或 open_clip 预训练标签。"""
        self.device = device

        # force_quick_gelu:仅 openai 原版权重需要;LAION 权重与默认 GELU 一致
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=weights or None, device=device,
            force_quick_gelu=(weights == "openai"))
        self.model = model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.preprocess = preprocess

        with torch.no_grad():
            self.temp = model.logit_scale.exp().float()  # CLIP 温度 ≈100

        v = model.visual
        # 仅支持 open_clip 标准 ViT 塔(resblocks 结构),B-16-plus-240 即此结构
        assert not hasattr(v, "trunk"), "timm-trunk 结构请换 open_clip 3.x"
        self.v = v
        self.blocks = v.transformer.resblocks
        self.grid = tuple(v.grid_size) if hasattr(v, "grid_size") else \
            (v.image_size[0] // v.patch_size[0], v.image_size[1] // v.patch_size[1])
        self.patch_dim = v.conv1.out_channels  # 896
        self.image_size = 240

        # 每类文本原型(整图 CLS 打分与窗口打分共用)
        self.normal_proto = None   # (1, 640) l2
        self.abnormal_proto = None
        self.gallery = None        # few-shot 参考特征 {large/mid/patch: (N, 640)}

    # ------------------------------------------------------------------
    # 冻结前向工具(全部 torch.no_grad 下调用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _patch_feats(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,240,240) → (B, 226, 896) conv+[CLS]+位置编码+ln_pre 的序列。"""
        v = self.v
        x = v.conv1(img).reshape(img.shape[0], self.patch_dim, -1).permute(0, 2, 1)
        cls = v.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                      # (B, 226, 896)
        x = x + v.positional_embedding
        return v.ln_pre(x)

    @torch.no_grad()
    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """token 序列过 12 层 transformer(任意子序列/窗口均可复用)。"""
        for blk in self.blocks:
            x = blk(x)
        return x

    @torch.no_grad()
    def _to_shared(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """token → ln_post → proj → 640 共享空间(与文本同空间),l2。"""
        h = self.v.ln_post(cls_tokens)
        return F.normalize(h @ self.v.proj, dim=-1)

    @torch.no_grad()
    def _window_indices(self, k: int) -> torch.Tensor:
        """15×15 网格上 kernel=k×k,stride=1 滑窗 → 每窗口的 patch token 索引。

        返回 (n_win, k*k) 的 token 序号(1..225,不含 CLS 的 0)。
        """
        board = torch.arange(1, self.grid[0] * self.grid[1] + 1,
                             dtype=torch.float32, device=self.device)
        board = board.view(1, 1, self.grid[0], self.grid[1])
        masks = F.unfold(board, kernel_size=k, stride=1).squeeze(0)  # (k², n_win)
        return masks.t().long()

    @torch.no_grad()
    def _win_feats(self, base: torch.Tensor, win_idx: torch.Tensor) -> torch.Tensor:
        """窗口子序列 [CLS+窗口 patches] 批量过塔 → 窗口级 CLS 特征 (n_win, 640)。

        base: 已 ln_pre 的整图序列 (1, 226, 896);ln_post → proj → 640 l2。
        """
        seq = torch.cat([torch.zeros(win_idx.shape[0], 1, dtype=torch.long,
                                     device=self.device), win_idx], dim=1)
        w = self._run_blocks(base[0][seq])               # (n_win, k²+1, 896)
        ln = F.normalize(self.v.ln_post(w[:, 0]), dim=-1)     # (n_win, 896)
        return F.normalize(ln @ self.v.proj, dim=-1)

    # ------------------------------------------------------------------
    # 文本原型
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_class(self, cls_name: str) -> None:
        """编码当前类的 normal/abnormal 原型(各 1 个向量,模板平均)。"""
        from prompts import build_class_prompts
        p = build_class_prompts(cls_name)

        def _proto(texts):
            tok = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tok)
            feats = F.normalize(feats.float(), dim=-1)
            return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)

        self.normal_proto = _proto(p["normal"])
        self.abnormal_proto = _proto(p["abnormal"])
        self.cls_name = cls_name

    # ------------------------------------------------------------------
    # few-shot 参考库(全权重冻结,仅缓存 k 张正常图的特征)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_gallery(self, imgs: torch.Tensor) -> None:
        """预计算 k 张参考正常图的三尺度特征到 640 共享空间(已 l2)。

        imgs: (k, 3, 240, 240) 已归一化张量。large/mid = 3×3/2×2 窗口 CLS,
        patch = 整图 12 层后逐 patch。与查询同空间,余弦近邻即可检索。
        """
        g = {"large": [], "mid": [], "patch": []}
        for i in range(imgs.shape[0]):
            feats = self._patch_feats(imgs[i:i + 1])
            z = self._run_blocks(feats)
            g["patch"].append(self._to_shared(z[0, 1:]))        # (225, 640)
            for name, k in (("large", 3), ("mid", 2)):
                win_idx = self._window_indices(k)
                g[name].append(self._win_feats(feats, win_idx))  # (n_win, 640)
        self.gallery = {name: torch.cat(v, dim=0) for name, v in g.items()}

    @staticmethod
    def _few_token_score(cur: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """(N,640) 查询 × (M,640) 参考 → 每 token 最近邻异常分 0.5·(1−max cos)。"""
        sim = cur @ mem.t()                        # (N, M)
        return 0.5 * (1.0 - sim.max(dim=-1)[0])

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    @staticmethod
    def _prob(feats: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
              temp: float) -> torch.Tensor:
        """(n, 640) 特征 × [pos,neg] → 温度缩放 softmax → 异常类概率 (n,)。"""
        logits = torch.cat([feats @ pos.t(), feats @ neg.t()], dim=1) * temp
        return torch.softmax(logits, dim=1)[:, 1]

    @staticmethod
    def _harmonic_to_patch(win_prob: torch.Tensor, win_idx: torch.Tensor,
                           grid_hw: int) -> torch.Tensor:
        """窗口级分数 → 每 patch 的调和平均(覆盖该 patch 的窗口)。

        A[p, w] = 窗口 w 是否覆盖 patch p;patch 分数 = Σ_w A / Σ_w (A / s_w)。
        """
        n_patch = grid_hw * grid_hw
        A = torch.zeros(win_idx.shape[0], n_patch,
                        dtype=win_prob.dtype, device=win_prob.device)
        A.scatter_(1, win_idx - 1, 1.0)       # win_idx 是 1..225 token 号
        cnt = A.sum(0)                            # 每 patch 被多少窗口覆盖
        inv = A.t() @ (1.0 / win_prob)            # Σ_w 1/s_w(仅覆盖窗口)
        return torch.nan_to_num(cnt / inv)        # (n_patch,)

    @torch.no_grad()
    def anomaly_maps(self, img_tensor: torch.Tensor,
                     use_few: bool = False) -> tuple:
        """(1,3,H,W) → (map (1,15,15), image_score)。

        zero-shot:map = 三路调和 48px/32px 窗口异常概率 + 整图 CLS 异常概率
        (逐 patch 广播);image 分 = CLS 异常概率。
        few-shot:窗口/patch 特征对 gallery 最近邻 0.5·(1−cos),窗口尺度调和
        到 patch,与 patch 尺度平均 → few map;map += few map,
        image = (文本概率 + max(few map))/2。
        """
        assert self.normal_proto is not None, "先 set_class()"
        if use_few:
            assert self.gallery is not None, "先 set_gallery()"
        pos, neg, temp = self.normal_proto, self.abnormal_proto, self.temp
        H, W = self.grid

        feats = self._patch_feats(img_tensor)          # (1, 226, 896)
        z = self._run_blocks(feats)                    # 整图 12 层
        cls_shared = self._to_shared(z[:, 0])          # (1, 640)
        cls_prob = self._prob(cls_shared, pos, neg, temp)  # (1,)

        # 多尺度窗口:3×3(48px,169 窗)与 2×2(32px,196 窗)批量过塔
        win_feats, maps = {}, []
        for name, k in (("large", 3), ("mid", 2)):
            win_idx = self._window_indices(k)          # (n_win, k*k)
            ws = self._win_feats(feats, win_idx)       # (n_win, 640)
            win_feats[name] = ws
            win_prob = self._prob(ws, pos, neg, temp)
            maps.append(self._harmonic_to_patch(win_prob, win_idx, H).view(H, W))

        # 三路调和(窗口两尺度 + 整图 CLS);CLS 项逐 patch 广播
        m48, m32 = maps
        m_all = 3.0 / (1.0 / m48 + 1.0 / m32 + 1.0 / cls_prob.item())
        m_all = torch.nan_to_num(m_all)

        if use_few:
            g = self.gallery
            patch_shared = self._to_shared(z[0, 1:])   # (225, 640)
            # 三尺度最近邻异常图:窗口尺度调和到 patch,补丁尺度直接 reshape
            few_maps = []
            for name, k in (("large", 3), ("mid", 2)):
                win_idx = self._window_indices(k)
                tok = self._few_token_score(win_feats[name], g[name])
                few_maps.append(self._harmonic_to_patch(tok, win_idx, H).view(H, W))
            patch_score = self._few_token_score(patch_shared, g["patch"]).view(H, W)
            few_map = (few_maps[0] + few_maps[1] + patch_score) / 3.0
            m_all = torch.nan_to_num(m_all + few_map)
            img_score = (cls_prob.item() + few_map.max().item()) / 2.0
        else:
            img_score = cls_prob.item()

        return m_all.unsqueeze(0).unsqueeze(0), img_score
