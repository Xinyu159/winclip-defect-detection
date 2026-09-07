"""WinCLIP 核心:CLIP patch 级特征 + 正常/异常文本双分支打分。

zero-shot:只用类别名文本,无任何训练/微调;
few-shot: 用 k 张正常样本的 patch 特征作参考,替换 normal 文本分支。

与论文一致的关键机制:
  1. 图像被切成 patch 窗口,ViT 的 patch token 保留空间位置(不是只有 CLS);
  2. 文本侧 = 状态词(正常/异常)× 句式模板 → 两类语义原型;
  3. 每 patch 对"异常原型最大相似度 - 正常原型相似度"→ 缺陷分数;
  4. image 级分数 = patch 分数图经 3×3 窗口投票后 top-k 池化(论文的窗口聚合思想)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import open_clip

_ARCH_TEMPLATE = "ViT-B-32"  # patch 32px,224 输入 → 7×7=49 patch


def _detect_patch_backend(visual) -> tuple:
    """定位产出 patch token 的前向路径,兼容 open_clip 不同版本结构。

    返回 (forward_fn, name):
      tims 版:  visual.trunk.blocks[-1](hook)→ 若存在 trunk.norm 则过 norm
      open_clip 官方版: visual.transformer.resblocks[-1](hook)→ ln_post
    """
    if hasattr(visual, "trunk"):  # open_clip >= 2.2x 部分版本用 timm trunk
        trunk = visual.trunk
        assert hasattr(trunk, "blocks"), f"trunk 无 blocks:{type(trunk)}"
        post_norm = getattr(trunk, "norm", None)
        return trunk.blocks[-1], post_norm, "timm-trunk"
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        post_norm = getattr(visual, "ln_post", None)
        return blocks[-1], post_norm, "openclip-resblocks"
    raise RuntimeError(f"无法识别 open_clip 视觉塔结构:{type(visual)}")


class WinCLIP:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 device: str = "cuda", image_size: int = 224,
                 mode: str = "diff", vote_win: int = 3, topk_pct: float = 0.05):
        self.device = device
        self.image_size = image_size
        self.mode = mode          # diff | softmax(消融用)
        self.vote_win = vote_win  # image 分数的窗口投票尺寸(论文 window 聚合)
        self.topk_pct = topk_pct  # image 分数 top-k 池化比例

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.model = model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.preprocess = preprocess

        # 温度:CLIP 文本-图像 logit 尺度(softmax 模式用)
        with torch.no_grad():
            self.temp = model.logit_scale.exp().float().item()

        # --- patch 特征 hook ---
        block, post_norm, backend_name = _detect_patch_backend(model.visual)
        self._post_norm = post_norm
        self._backend_name = backend_name
        self._hook_out = None

        def _hook(module, args, out):
            self._hook_out = out.detach()

        self._handle = block.register_forward_hook(_hook)
        self._patch_side = None  # 探测后缓存

    # ------------------------------------------------------------------
    def encode_patches(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,N,D) patch tokens(不含 CLS,行 l2 归一)。"""
        self._hook_out = None
        self.model.encode_image(img_tensor)  # 前向,触发 hook
        h = self._hook_out
        if h is None:
            raise RuntimeError("patch hook 未触发,结构探测失败")
        if self._post_norm is not None:
            h = self._post_norm(h)
        h = h[:, 1:]  # 去 CLS
        return F.normalize(h.float(), dim=-1)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tok)
        return F.normalize(feats.float(), dim=-1)

    # ------------------------------------------------------------------
    def set_class(self, cls_noun: str) -> None:
        """编码当前类的正常/异常文本原型。

        normal:   各状态词(模板已平均)→ (k, D)
        abnormal: 各状态词(模板已平均)→ (m, D)
        """
        from prompts import TEMPLATES, build_class_prompts
        p = build_class_prompts(cls_noun)
        n_tpl = len(TEMPLATES)
        with torch.no_grad():
            self.normal_states = self._state_vectors(p["normal"], n_tpl)
            self.abnormal_states = self._state_vectors(p["abnormal"], n_tpl)
        self._prompt_info = {"class": cls_noun,
                             "n_prompts": len(p["normal"]),
                             "a_prompts": len(p["abnormal"])}
        self.ref_feats = None  # few-shot 注入时赋值

    def _state_vectors(self, prompt_list: list[str], n_templates: int) -> torch.Tensor:
        """prompt 按 (状态词 × 模板) 排布 → 每状态词模板平均得一个向量。"""
        feats = self.encode_texts(prompt_list)             # (S*T, D)
        return feats.reshape(-1, n_templates, feats.shape[-1]).mean(dim=1)

    # ------------------------------------------------------------------
    def _similarities(self, patch_feats: torch.Tensor, states: torch.Tensor,
                      reduce: str = "mean") -> torch.Tensor:
        """patch(N,D) × states(S,D) → (N,) 相似度(状态词内模板已平均)。

        注:为保持 zero-shot 与 few-shot 计算一致,相似度不用温度缩放,
        直接余弦;ROC 排序对单调变换不变。softmax 模式才引入温度。
        """
        sim = patch_feats @ states.t()  # (N, S)
        if reduce == "mean":
            return sim.mean(dim=1)
        if reduce == "max":
            return sim.max(dim=1).values
        raise ValueError(reduce)

    @torch.no_grad()
    def anomaly_maps(self, img_tensor: torch.Tensor) -> tuple:
        """(1,3,H,W) → (patch_map(N,), image_score) 。few-shot 时 normal
        分支用参考 patch 特征(nearest-neighbor 相似度)。"""
        patch = self.encode_patches(img_tensor)  # (N, D) l2
        if self.ref_feats is None:
            sim_n = self._similarities(patch, self.normal_states, "mean")
        else:  # few-shot:与 k-shot 正常参考 patch 集的最大相似度
            sim_n = (patch @ self.ref_feats.t()).max(dim=1).values
        sim_a = self._similarities(patch, self.abnormal_states, "max")

        if self.mode == "diff":
            p_map = sim_a - sim_n                      # 线性差分,可负
        elif self.mode == "softmax":
            # 状态级 softmax:logits 已 l2 余弦 + 温度 → P(异常状态) 之和
            logits = torch.cat([sim_n.unsqueeze(1), sim_a.unsqueeze(1)], dim=1) * self.temp
            p_map = torch.softmax(logits, dim=1)[:, 1]
        else:
            raise ValueError(self.mode)

        # image 级:窗口投票(3×3 均值,边界保持)+ top-k 池化
        side = int(patch.shape[0] ** 0.5)
        grid = p_map.view(1, 1, side, side)
        if self.vote_win > 1:
            grid = F.avg_pool2d(grid, self.vote_win, stride=1,
                                padding=self.vote_win // 2, count_include_pad=False)
        flat = grid.flatten()
        k = max(1, int(flat.numel() * self.topk_pct))
        image_score = flat.topk(k).values.mean().item()
        return p_map, image_score

    def close(self):
        self._handle.remove()
