"""
models/mask_rcnn.py
====================
基于 torchvision 的 Mask R-CNN 实例分割模型封装。

核心修改（相对于 torchvision 默认实现）：
  1. 类别数替换为冰山数据集的 num_classes（背景 + 冰山）
  2. Anchor 尺寸根据 SAR 切片中冰山的典型尺度调整
  3. 支持从配置文件动态设置所有 RPN/ROI 超参数
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


def build_mask_rcnn(cfg) -> MaskRCNN:
    """
    根据配置构建 Mask R-CNN 模型。

    Args:
        cfg: OmegaConf 配置对象（读取 cfg.model.mask_rcnn）

    Returns:
        torchvision MaskRCNN 实例，可直接调用：
          - 训练模式: model(images, targets) → loss_dict
          - 推理模式: model(images)          → List[prediction_dict]
    """
    mc = cfg.model.mask_rcnn      # 配置子节点快捷引用
    num_classes = int(mc.num_classes)  # 通常为 2：背景 + 冰山

    # ────────────────────────────────────
    # Step 1：自定义 Anchor 生成器
    #   默认 FPN 的 Anchor 尺度针对自然图像，
    #   SAR 冰山尺度差异大（几十到数千像素），需要调整。
    # ────────────────────────────────────
    anchor_sizes   = tuple(tuple(s) for s in mc.rpn_anchor_sizes)
    aspect_ratios  = tuple(tuple(r) for r in mc.rpn_aspect_ratios) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    # ────────────────────────────────────
    # Step 2：加载预训练骨干 + FPN
    # ────────────────────────────────────
    weights = "DEFAULT" if mc.pretrained_backbone else None

    # 自定义 Mask ROI Pooler，允许通过 config 控制分辨率
    # 默认 14→28：对 5-25px 小冰山，分辨率翻倍可显著提升 mask IoU
    mask_pool_size = int(mc.get("mask_roi_pool_output_size", 14))
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=mask_pool_size,
        sampling_ratio=2,
    )

    backbone_name = mc.get("backbone", "resnet50")
    if backbone_name == "resnet50":
        model = maskrcnn_resnet50_fpn(
            weights=None,           # 不加载完整 Mask R-CNN 权重
            weights_backbone=weights,
            min_size=int(mc.min_size),
            max_size=int(mc.max_size),
            # RPN 参数
            rpn_anchor_generator=anchor_generator,
            rpn_nms_thresh=float(mc.rpn_nms_thresh),
            rpn_fg_iou_thresh=float(mc.rpn_fg_iou_thresh),
            rpn_bg_iou_thresh=float(mc.rpn_bg_iou_thresh),
            rpn_batch_size_per_image=int(mc.rpn_batch_size_per_image),
            rpn_positive_fraction=float(mc.rpn_positive_fraction),
            rpn_pre_nms_top_n_train=int(mc.rpn_pre_nms_top_n_train),
            rpn_post_nms_top_n_train=int(mc.rpn_post_nms_top_n_train),
            rpn_pre_nms_top_n_test=int(mc.rpn_pre_nms_top_n_test),
            rpn_post_nms_top_n_test=int(mc.rpn_post_nms_top_n_test),
            # ROI Box 参数
            box_score_thresh=float(mc.box_score_thresh),
            box_nms_thresh=float(mc.box_nms_thresh),
            box_detections_per_img=int(mc.box_detections_per_img),
            box_fg_iou_thresh=float(mc.box_fg_iou_thresh),
            box_bg_iou_thresh=float(mc.box_bg_iou_thresh),
            box_batch_size_per_image=int(mc.box_batch_size_per_image),
            box_positive_fraction=float(mc.box_positive_fraction),
            # 自定义 Mask ROI Pooler
            mask_roi_pool=mask_roi_pool,
            # 类别数（含背景）
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"不支持的骨干网络: {backbone_name}，目前支持 'resnet50'。"
        )

    # ────────────────────────────────────
    # Step 3：替换 Box Head（分类 + 回归）
    #   torchvision 默认是 91 类（COCO），需要替换为 num_classes。
    # ────────────────────────────────────
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features_box,
        num_classes=num_classes,
    )

    # ────────────────────────────────────
    # Step 4：替换 Mask Head
    # ────────────────────────────────────
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256  # Mask Head 中间层通道数
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes,
    )

    # ────────────────────────────────────
    # Step 5：加载完整预训练权重（可选）
    # ────────────────────────────────────
    if mc.pretrained_weights is not None:
        state = torch.load(mc.pretrained_weights, map_location="cpu", weights_only=False)
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state, strict=False)

    return model


def get_mask_rcnn_params(model: MaskRCNN, lr: float, weight_decay: float):
    """
    为 Mask R-CNN 构建差异化学习率参数组：
      - 骨干层（backbone）使用较小学习率（× 0.1），避免破坏预训练特征
      - FPN + Head 使用标准学习率

    Args:
        model:        MaskRCNN 实例
        lr:           基准学习率（来自 config）
        weight_decay: 权重衰减

    Returns:
        适合传入 optimizer 的 param_groups 列表
    """
    backbone_params = []
    other_params    = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": backbone_params, "lr": lr * 0.1,  "weight_decay": weight_decay},
        {"params": other_params,    "lr": lr,         "weight_decay": weight_decay},
    ]
