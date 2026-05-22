"""
models/mask2former.py
======================
Mask2Former 实例分割模型封装（基于 HuggingFace transformers）。

支持骨干：
  swin_b → facebook/mask2former-swin-base-coco-instance  (~200M 参数)
  swin_l → facebook/mask2former-swin-large-coco-instance (~300M 参数)

训练接口与 torchvision Mask R-CNN 对齐：
  训练: forward(images, targets) → loss_dict  （sum of values = total loss）
  推理: forward(images)          → List[Dict[boxes/labels/masks/scores]]

依赖：
  pip install transformers>=4.38.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone 名称 → HuggingFace pretrained model ID
_HF_MODEL_MAP = {
    "swin_b": "facebook/mask2former-swin-base-coco-instance",
    "swin_l": "facebook/mask2former-swin-large-coco-instance",
}


def _import_transformers():
    """延迟导入 transformers，避免未安装时污染其他模型的加载。"""
    try:
        from transformers import (
            Mask2FormerConfig,
            Mask2FormerForUniversalSegmentation,
        )
        return Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    except ImportError:
        raise ImportError(
            "使用 mask2former 架构需要安装 transformers：\n"
            "  pip install transformers>=4.38.0"
        )


class Mask2FormerWrapper(nn.Module):
    """
    Mask2Former 包装器，对齐 torchvision Mask R-CNN 接口。

    关键设计：
      1. 输入 SAR [0,1]（单通道复制成 3 通道由 Dataset 完成）→ 内部 ImageNet 归一化
      2. 训练目标 labels 为 1-indexed（torchvision 约定），内部转为 0-indexed（HF 约定）
      3. 损失字典仅含 "loss" 一项，使 sum(loss_dict.values()) == total_loss
         （若含分量损失会造成梯度重复累加）
    """

    def __init__(
        self,
        hf_model,
        score_threshold: float = 0.05,
        mask_threshold: float = 0.5,
        normalize_imagenet: bool = True,
    ):
        super().__init__()
        self.model = hf_model
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        # ImageNet 归一化参数（Swin 骨干以 ImageNet 均值/标准差预训练）
        if normalize_imagenet:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            mean = torch.zeros(1, 3, 1, 1)
            std  = torch.ones(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std",  std)

    # ── 前向传播 ──────────────────────────────────────────────────────

    def forward(self, images, targets=None):
        """
        images:  List[Tensor(C, H, W)]  来自 DataLoader collate_fn
        targets: List[Dict]             训练时提供（torchvision 格式）
        """
        pixel_values = torch.stack(images)                      # (B, C, H, W)
        pixel_values = (pixel_values - self._imagenet_mean) / self._imagenet_std

        if self.training and targets is not None:
            return self._forward_train(pixel_values, targets)
        return self._forward_inference(pixel_values)

    # ── 训练前向 ──────────────────────────────────────────────────────

    def _forward_train(self, pixel_values, targets):
        mask_labels  = []
        class_labels = []
        for t in targets:
            mask_labels.append(t["masks"].float())      # (N, H, W)
            class_labels.append((t["labels"] - 1).long())  # 1-indexed → 0-indexed

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        # 仅返回总损失，避免分量损失被重复求和
        return {"loss": outputs.loss}

    # ── 推理前向 ──────────────────────────────────────────────────────

    def _forward_inference(self, pixel_values):
        B, C, H, W = pixel_values.shape
        outputs = self.model(pixel_values=pixel_values)

        # class_queries_logits: (B, num_queries, num_labels + 1)  最后一维 = void
        # masks_queries_logits: (B, num_queries, H//4, W//4)
        class_logits = outputs.class_queries_logits   # (B, Q, K+1)
        masks_logits = outputs.masks_queries_logits   # (B, Q, h, w)

        results = []
        for b in range(B):
            # 前景类置信度（排除 void 类）
            scores_per_cls = class_logits[b, :, :-1].softmax(dim=-1)   # (Q, K)
            pred_scores, pred_classes = scores_per_cls.max(dim=-1)     # (Q,)

            keep = pred_scores > self.score_threshold
            pred_scores  = pred_scores[keep]
            pred_classes = pred_classes[keep] + 1   # 0-indexed → 1-indexed（torchvision 约定）

            msk_b = masks_logits[b][keep]           # (N, h, w)
            N = msk_b.shape[0]

            if N > 0:
                # 上采样到原始分辨率（Mask2Former 输出为 1/4 分辨率）
                msk_up = F.interpolate(
                    msk_b.unsqueeze(1).float(),     # (N, 1, h, w)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )                                   # (N, 1, H, W)
                pred_masks = torch.sigmoid(msk_up)  # float 概率图
                binary_masks = (pred_masks.squeeze(1) > self.mask_threshold)  # (N, H, W)
                boxes = _masks_to_boxes(binary_masks)
            else:
                pred_masks = torch.zeros((0, 1, H, W), device=pixel_values.device)
                boxes      = torch.zeros((0, 4),       device=pixel_values.device)

            results.append({
                "boxes":  boxes,
                "labels": pred_classes,
                "masks":  pred_masks,
                "scores": pred_scores,
            })

        return results


# ── 工具函数 ──────────────────────────────────────────────────────────

def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """(N, H, W) bool → (N, 4) xyxy float"""
    if masks.shape[0] == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float32)
    for i, m in enumerate(masks):
        ys, xs = torch.where(m)
        if xs.numel() == 0:
            continue
        boxes[i] = torch.stack(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        ).float()
    return boxes


# ── 工厂函数 ──────────────────────────────────────────────────────────

def build_mask2former(cfg) -> Mask2FormerWrapper:
    """
    根据配置构建 Mask2Former 封装模型。

    Args:
        cfg: OmegaConf 配置对象（读取 cfg.model.mask2former）

    Returns:
        Mask2FormerWrapper（未移动到设备，由调用方 .to(device)）

    网络访问说明：
      - 中国大陆服务器（AutoDL 等）默认使用 hf-mirror.com 镜像，无需额外配置
      - 完全离线环境：在 configs/config.yaml 中设置
          model.mask2former.local_model_path: /path/to/local/model
        本地模型目录须包含 config.json 和 pytorch_model.bin（或 model.safetensors）
      - 手动下载命令（需要网络，在有网的机器上执行）：
          pip install huggingface_hub
          huggingface-cli download facebook/mask2former-swin-base-coco-instance --local-dir /root/autodl-tmp/mask2former-swin-b
          huggingface-cli download facebook/mask2former-swin-large-coco-instance --local-dir /root/autodl-tmp/mask2former-swin-l
    """
    import os
    Mask2FormerConfig, Mask2FormerForUniversalSegmentation = _import_transformers()

    # 中国大陆服务器默认使用镜像（HF_ENDPOINT 未设置时才覆盖，已设置则保持用户配置）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    mc = cfg.model.mask2former
    backbone_variant = str(mc.backbone)

    if backbone_variant not in _HF_MODEL_MAP:
        raise ValueError(
            f"不支持的骨干: '{backbone_variant}'。"
            f"支持: {list(_HF_MODEL_MAP.keys())}"
        )

    hf_model_name  = _HF_MODEL_MAP[backbone_variant]
    num_labels     = int(mc.get("num_labels", 1))
    # local_model_path 优先：指向本地已下载的模型目录，完全绕过网络请求
    local_path     = mc.get("local_model_path", None)
    model_source   = str(local_path) if local_path else hf_model_name

    # id2label / label2id 仅含前景类，不含背景（HF Mask2Former 约定）
    id2label = {i: f"class_{i}" for i in range(num_labels)}
    id2label[0] = "iceberg"
    label2id = {v: k for k, v in id2label.items()}

    if mc.pretrained_backbone:
        # 从 COCO 预训练权重加载骨干 + 解码器，仅分类头因 num_labels 不同而重新初始化
        config = Mask2FormerConfig.from_pretrained(model_source)
        config.num_labels = num_labels
        config.id2label   = id2label
        config.label2id   = label2id
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_source,
            config=config,
            ignore_mismatched_sizes=True,   # 分类头维度变更，允许不匹配
        )
    else:
        # 仅使用架构，不加载预训练权重（仍需 config.json 获取网络结构）
        config = Mask2FormerConfig.from_pretrained(model_source)
        config.num_labels = num_labels
        config.id2label   = id2label
        config.label2id   = label2id
        hf_model = Mask2FormerForUniversalSegmentation(config)

    wrapper = Mask2FormerWrapper(
        hf_model=hf_model,
        score_threshold=float(mc.get("score_threshold", 0.05)),
        mask_threshold=float(mc.get("mask_threshold", 0.5)),
        normalize_imagenet=bool(mc.get("normalize_imagenet", True)),
    )

    return wrapper


def get_mask2former_params(model: Mask2FormerWrapper, lr: float, weight_decay: float):
    """
    差异化学习率参数组：
      - Swin 骨干（pixel_level_module.encoder）: lr × 0.1
      - 像素解码器 + Transformer 解码器 + 分类/掩膜头: 标准 lr
    """
    backbone_params = []
    other_params    = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "pixel_level_module.encoder" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": backbone_params, "lr": lr * 0.1, "weight_decay": weight_decay},
        {"params": other_params,    "lr": lr,        "weight_decay": weight_decay},
    ]
