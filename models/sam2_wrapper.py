"""
models/sam2_wrapper.py
======================
SAM2 (Segment Anything Model 2) 封装，用于冰山实例分割。

训练策略：GT 边界框作为提示词 → 优化掩膜解码器（+ 可选图像编码器）
推理策略：SAM2AutomaticMaskGenerator，格式化为 Mask R-CNN 风格输出

支持变体（hiera 骨干）：
  tiny       ~39M 参数
  small      ~46M 参数
  base_plus  ~80M 参数
  large      ~224M 参数

依赖：
  pip install git+https://github.com/facebookresearch/sam2.git
  # 或（SAM2.1）：
  pip install git+https://github.com/facebookresearch/sam2.git@main

权重下载（在服务器上执行）：
  # SAM2.1（推荐）
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  # SAM2.0（备用）
  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 变体 → 配置文件名映射 ──────────────────────────────────────────

_V2_CFG = {
    "tiny":      "sam2_hiera_tiny.yaml",
    "small":     "sam2_hiera_small.yaml",
    "base_plus": "sam2_hiera_base_plus.yaml",
    "large":     "sam2_hiera_large.yaml",
}
_V2_CKPT = {
    "tiny":      "sam2_hiera_tiny.pt",
    "small":     "sam2_hiera_small.pt",
    "base_plus": "sam2_hiera_base_plus.pt",
    "large":     "sam2_hiera_large.pt",
}
_V21_CFG = {
    "tiny":      "sam2.1_hiera_tiny.yaml",
    "small":     "sam2.1_hiera_small.yaml",
    "base_plus": "sam2.1_hiera_base_plus.yaml",
    "large":     "sam2.1_hiera_large.yaml",
}
_V21_CKPT = {
    "tiny":      "sam2.1_hiera_tiny.pt",
    "small":     "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large":     "sam2.1_hiera_large.pt",
}


def _import_sam2():
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        return build_sam2, SAM2ImagePredictor, SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "使用 sam2 架构需要安装 SAM2：\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )


# ══════════════════════════════════════════════════════════════════
# SAM2 封装模型
# ══════════════════════════════════════════════════════════════════

class SAM2Wrapper(nn.Module):
    """
    SAM2 实例分割封装，接口对齐 Mask R-CNN / Mask2Former / YOLO。

    训练模式（GT box 提示）：
      forward(images, targets) → {"loss_seg": Tensor, "loss_score": Tensor}

    推理模式（自动掩膜生成）：
      forward(images) → List[Dict[boxes/labels/scores/masks]]

    设计要点：
      - 训练时直接调用 sam_prompt_encoder + sam_mask_decoder（梯度可回传）
      - 图像编码器默认冻结（freeze_image_encoder=true），节省显存
      - 推理时使用 SAM2AutomaticMaskGenerator 密集点提示，无需手工框
      - 损失 = BCE（pos_weight=10）× 0.5 + Dice × 0.5 + IOU评分损失 × 0.05
    """

    def __init__(self, predictor, auto_gen_kwargs: dict, cfg):
        super().__init__()
        self.predictor = predictor

        sc = cfg.model.sam2
        self.score_threshold      = float(sc.score_threshold)
        self.mask_threshold       = float(sc.mask_threshold)
        self.freeze_image_encoder = bool(sc.freeze_image_encoder)
        self._multimask_output    = bool(sc.get("multimask_output", False))
        self._auto_gen_kwargs     = dict(auto_gen_kwargs)

        if self.freeze_image_encoder:
            for p in predictor.model.image_encoder.parameters():
                p.requires_grad_(False)

        # pos_weight 作为 buffer 跟随 .to(device) 移动
        self.register_buffer("_pos_weight", torch.tensor([10.0]))

    # ── train()/eval() 切换时同步内部模型状态 ─────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.predictor.model.sam_mask_decoder.train(True)
            self.predictor.model.sam_prompt_encoder.train(True)
            if not self.freeze_image_encoder:
                self.predictor.model.image_encoder.train(True)
            else:
                self.predictor.model.image_encoder.eval()
        else:
            self.predictor.model.eval()
        return self

    # ── 接口分发 ──────────────────────────────────────────────────

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self._forward_train(images, targets)
        return self._forward_inference(images)

    # ══════════════════════════════════════════════════════════════
    # 训练前向：GT box 作为提示词
    # ══════════════════════════════════════════════════════════════

    def _forward_train(self, images, targets):
        device = next(self.predictor.model.parameters()).device

        loss_seg_sum   = torch.tensor(0.0, device=device)
        loss_score_sum = torch.tensor(0.0, device=device)
        n_instances = 0

        for image, target in zip(images, targets):
            boxes    = target["boxes"]    # (N, 4) xyxy float, on device
            gt_masks = target["masks"]   # (N, H, W) bool, on device
            N = len(boxes)
            if N == 0:
                continue

            # (3, H, W) float [0,1] → numpy (H, W, 3) uint8
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # set_image 缓存图像特征（梯度在 freeze=True 时不经过编码器）
            self.predictor.set_image(img_np)

            # _prep_prompts：将 box 坐标转为 SAM2 归一化空间，返回 point 格式
            # input_box: (N, 4) numpy xyxy；返回 unnorm_coords (N,2,2), labels (N,2)
            boxes_np = boxes.detach().cpu().numpy()
            _, unnorm_coords, labels, _ = self.predictor._prep_prompts(
                None, None,
                box=boxes_np,
                mask_logits=None,
                normalize_coords=True,
            )

            # 提示编码器（SAM2 将 box 角点视为 label=2/3 的特殊点）
            sparse_emb, dense_emb = self.predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )

            # 高分辨率特征（供掩膜解码器使用）
            high_res = [
                feat[-1].unsqueeze(0)
                for feat in self.predictor._features["high_res_feats"]
            ]

            # 掩膜解码器
            # repeat_image=True：解码器内部将图像嵌入从 (1,C,H,W) 扩展到 (N,C,H,W)
            low_res_masks, iou_preds, _, _ = self.predictor.model.sam_mask_decoder(
                image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=self._multimask_output,
                repeat_image=(N > 1),
                high_res_features=high_res,
            )
            # low_res_masks: (N, 1, 256, 256) 原始 logits
            # iou_preds:     (N, 1)

            # 上采样到原始分辨率，保留梯度
            pred_masks_full = self.predictor._transforms.postprocess_masks(
                low_res_masks, self.predictor._orig_hw[-1],
            )
            # pred_masks_full: (N, 1, H, W) logits

            pred_logits = pred_masks_full[:, 0]   # (N, H, W)
            pred_iou    = iou_preds[:, 0]          # (N,)

            gt = gt_masks.float().to(device)       # (N, H, W)

            # BCE + Dice（参考 U-Net 损失，对小目标友好）
            bce = F.binary_cross_entropy_with_logits(
                pred_logits, gt,
                pos_weight=self._pos_weight,
                reduction="mean",
            )
            p    = torch.sigmoid(pred_logits)
            inter = (p * gt).sum(dim=(1, 2))
            dice_per = 1.0 - (2.0 * inter + 1e-6) / (
                p.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) + 1e-6
            )
            seg_loss = bce * 0.5 + dice_per.mean() * 0.5

            # IOU 评分损失：让模型自身置信度对齐实际 IoU
            with torch.no_grad():
                true_iou = _iou_batch(p > 0.5, gt > 0.5)   # (N,)
            score_loss = torch.abs(pred_iou - true_iou).sum()

            loss_seg_sum   = loss_seg_sum   + seg_loss * N
            loss_score_sum = loss_score_sum + score_loss
            n_instances   += N

        if n_instances == 0:
            # 空批次（所有 target 均无 GT），返回零损失且保持梯度图不断
            dummy = sum(
                p.sum() * 0.0
                for p in self.predictor.model.parameters()
                if p.requires_grad
            )
            return {"loss_seg": dummy, "loss_score": dummy}

        return {
            "loss_seg":   loss_seg_sum   / n_instances,
            "loss_score": loss_score_sum / n_instances * 0.05,
        }

    # ══════════════════════════════════════════════════════════════
    # 推理前向：自动掩膜生成
    # ══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _forward_inference(self, images):
        _, _, SAM2AutomaticMaskGenerator = _import_sam2()

        gen = SAM2AutomaticMaskGenerator(
            model=self.predictor.model,
            **self._auto_gen_kwargs,
        )

        results = []
        for image in images:
            _, H, W = image.shape
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # ann: list of dicts，每个 dict 含:
            #   segmentation  (H, W) bool
            #   stability_score  float
            #   predicted_iou    float
            #   bbox             [x, y, w, h]
            ann_list = gen.generate(img_np)

            keep = [
                a for a in ann_list
                if a["stability_score"] >= self.score_threshold
            ]

            if not keep:
                results.append({
                    "boxes":  torch.zeros(0, 4,    device=image.device),
                    "labels": torch.zeros(0,       device=image.device, dtype=torch.int64),
                    "scores": torch.zeros(0,       device=image.device),
                    "masks":  torch.zeros(0, 1, H, W, device=image.device),
                })
                continue

            scores  = torch.tensor(
                [a["stability_score"] for a in keep],
                device=image.device, dtype=torch.float32,
            )
            labels  = torch.ones(len(keep), device=image.device, dtype=torch.int64)

            # segmentation 掩膜堆叠
            mask_np = np.stack([a["segmentation"] for a in keep])    # (N, H, W) bool
            masks_t = torch.from_numpy(mask_np).unsqueeze(1).float().to(image.device)

            # bbox: SAM2 返回 xywh → 转为 xyxy
            bboxes = [
                [a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2],
                 a["bbox"][1] + a["bbox"][3]]
                for a in keep
            ]
            boxes_t = torch.tensor(bboxes, device=image.device, dtype=torch.float32)

            results.append({
                "boxes":  boxes_t,
                "labels": labels,
                "scores": scores,
                "masks":  masks_t,
            })

        return results


# ── 工具函数 ──────────────────────────────────────────────────────

def _iou_batch(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> torch.Tensor:
    """(N, H, W) bool → (N,) IoU，用于 IOU 评分损失。"""
    inter = (pred_bin & gt_bin).sum(dim=(1, 2)).float()
    union = (pred_bin | gt_bin).sum(dim=(1, 2)).float()
    return inter / (union + 1e-6)


# ══════════════════════════════════════════════════════════════════
# 工厂函数 & 参数组
# ══════════════════════════════════════════════════════════════════

def build_sam2_model(cfg) -> SAM2Wrapper:
    """
    根据配置构建 SAM2 封装模型。

    本地权重放置示例（AutoDL）：
      /root/autodl-tmp/sam2.1_hiera_tiny.pt
    然后在 config.yaml 中设置：
      model.sam2.checkpoint_path: /root/autodl-tmp/sam2.1_hiera_tiny.pt
    """
    build_sam2, SAM2ImagePredictor, _ = _import_sam2()

    sc      = cfg.model.sam2
    variant = str(sc.variant).lower()
    version = str(sc.get("version", "2.1"))

    if version == "2.1":
        cfg_map, ckpt_map = _V21_CFG, _V21_CKPT
    else:
        cfg_map, ckpt_map = _V2_CFG,  _V2_CKPT

    if variant not in cfg_map:
        raise ValueError(
            f"不支持的 SAM2 变体: '{variant}'。"
            f"支持: {list(cfg_map.keys())}"
        )

    model_cfg_name = cfg_map[variant]
    ckpt_path      = sc.get("checkpoint_path", None)

    if not ckpt_path:
        # 尝试 SAM2 安装目录下的 checkpoints 子目录
        try:
            import sam2 as _sam2_pkg
            pkg_root  = os.path.dirname(os.path.dirname(_sam2_pkg.__file__))
            candidate = os.path.join(pkg_root, "checkpoints", ckpt_map[variant])
            if os.path.exists(candidate):
                ckpt_path = candidate
        except Exception:
            pass

    if not ckpt_path:
        # 最后 fallback：传 None，build_sam2 会尝试自动下载或报错
        ckpt_path = None

    print(f"[SAM2] 版本: {version}  变体: {variant}  配置: {model_cfg_name}")
    print(f"[SAM2] 权重路径: {ckpt_path or '(未指定，尝试自动下载)'}")

    sam2_model = build_sam2(model_cfg_name, ckpt_path, device="cpu")
    predictor  = SAM2ImagePredictor(sam2_model)

    auto_gen_kwargs = {
        "points_per_side":        int(sc.get("points_per_side",        32)),
        "pred_iou_thresh":        float(sc.get("pred_iou_thresh",       0.70)),
        "stability_score_thresh": float(sc.get("stability_score_thresh", 0.85)),
        "box_nms_thresh":         float(sc.get("box_nms_thresh",        0.70)),
    }

    return SAM2Wrapper(predictor=predictor, auto_gen_kwargs=auto_gen_kwargs, cfg=cfg)


def get_sam2_params(model: SAM2Wrapper, lr: float, weight_decay: float) -> list:
    """
    差异化学习率参数组：
      - 图像编码器（冻结时不出现）：lr × 0.01
      - 提示编码器：                 lr × 0.1
      - 掩膜解码器：                 标准 lr
    """
    img_enc_ids = {id(p) for p in model.predictor.model.image_encoder.parameters()}
    pmt_enc_ids = {id(p) for p in model.predictor.model.sam_prompt_encoder.parameters()}

    img_params  = []
    pmt_params  = []
    dec_params  = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in img_enc_ids:
            img_params.append(p)
        elif pid in pmt_enc_ids:
            pmt_params.append(p)
        else:
            dec_params.append(p)

    groups = []
    if img_params:
        groups.append({"params": img_params,  "lr": lr * 0.01, "weight_decay": weight_decay})
    if pmt_params:
        groups.append({"params": pmt_params,  "lr": lr * 0.1,  "weight_decay": weight_decay})
    groups.append(    {"params": dec_params,   "lr": lr,         "weight_decay": weight_decay})

    return groups
