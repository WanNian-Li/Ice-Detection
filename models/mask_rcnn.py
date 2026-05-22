"""
models/mask_rcnn.py
====================
基于 torchvision 的 Mask R-CNN 实例分割模型封装。

支持两种骨干接入方式：
  1. torchvision ResNet50 + FPN（默认，backbone: "resnet50"）
  2. timm 任意骨干 + FPN（策略B，backbone: "convnext_small.fb_in22k_ft_in1k" 等）
     timm 骨干通过 ImageNet-22k 预训练，泛化能力优于 ImageNet-1k 的 ResNet50；
     ConvNeXt / Swin 的注意力/局部感受野对 SAR 斑点纹理理解更强。
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# timm 骨干 + FPN 封装
# ──────────────────────────────────────────────────────────────────

class TimmBackboneWithFPN(nn.Module):
    """
    将任意 timm 骨干（ConvNeXt / Swin 等）包装为 Mask R-CNN 可用的骨干+FPN。

    骨干输出 4 级特征（out_indices=0,1,2,3，stride=4/8/16/32），
    经 FPN 统一到 out_channels=256，再加 LastLevelMaxPool 得第 5 级（stride=64）。
    共 5 级与 rpn_anchor_sizes 中的 5 组 anchor 一一对应。

    推荐骨干（config backbone 字段直接填 timm 模型名）：
      - "convnext_small.fb_in22k_ft_in1k"   ConvNeXt-S，ImageNet-22k 预训练
      - "convnext_base.fb_in22k_ft_in1k"    ConvNeXt-B，精度更高，显存更多
      - "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"  Swin-T，注意力骨干
      - "swin_small_patch4_window7_224.ms_in22k_ft_in1k" Swin-S，精度/速度均衡
    """

    def __init__(self, backbone_name: str, pretrained: bool, out_channels: int = 256):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm 未安装。请运行 `pip install timm` 后重试。"
            )

        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        in_channels_list = self.body.feature_info.channels()  # e.g. [96,192,384,768]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        features  = self.body(x)                               # list of 4 tensors
        feat_dict = OrderedDict([(str(i), f) for i, f in enumerate(features)])
        return self.fpn(feat_dict)                             # dict: "0"–"3" + "pool"


# ──────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────

def build_mask_rcnn(cfg) -> MaskRCNN:
    """
    根据配置构建 Mask R-CNN 模型。

    backbone="resnet50"  → 使用 torchvision 内置实现（现有路径）
    backbone=其他字符串  → 视为 timm 模型名，走 TimmBackboneWithFPN 路径
    """
    mc = cfg.model.mask_rcnn
    num_classes = int(mc.num_classes)

    # ── Anchor 生成器（5 级，与 FPN 输出对齐）──
    anchor_sizes  = tuple(tuple(s) for s in mc.rpn_anchor_sizes)
    aspect_ratios = tuple(tuple(r) for r in mc.rpn_aspect_ratios) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # ── 自定义 Mask ROI Pooler ──
    mask_pool_size = int(mc.get("mask_roi_pool_output_size", 14))
    mask_roi_pool  = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=mask_pool_size,
        sampling_ratio=2,
    )

    backbone_name = str(mc.get("backbone", "resnet50"))
    pretrained    = bool(mc.pretrained_backbone)

    if backbone_name == "resnet50":
        # ── 路径 A：torchvision 内置 ResNet50 + FPN ──
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone="DEFAULT" if pretrained else None,
            min_size=int(mc.min_size),
            max_size=int(mc.max_size),
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
            box_score_thresh=float(mc.box_score_thresh),
            box_nms_thresh=float(mc.box_nms_thresh),
            box_detections_per_img=int(mc.box_detections_per_img),
            box_fg_iou_thresh=float(mc.box_fg_iou_thresh),
            box_bg_iou_thresh=float(mc.box_bg_iou_thresh),
            box_batch_size_per_image=int(mc.box_batch_size_per_image),
            box_positive_fraction=float(mc.box_positive_fraction),
            mask_roi_pool=mask_roi_pool,
            num_classes=num_classes,
        )

    else:
        # ── 路径 B：timm 骨干 + FPN（Strategy B）──
        backbone = TimmBackboneWithFPN(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            min_size=int(mc.min_size),
            max_size=int(mc.max_size),
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
            box_score_thresh=float(mc.box_score_thresh),
            box_nms_thresh=float(mc.box_nms_thresh),
            box_detections_per_img=int(mc.box_detections_per_img),
            box_fg_iou_thresh=float(mc.box_fg_iou_thresh),
            box_bg_iou_thresh=float(mc.box_bg_iou_thresh),
            box_batch_size_per_image=int(mc.box_batch_size_per_image),
            box_positive_fraction=float(mc.box_positive_fraction),
            mask_roi_pool=mask_roi_pool,
        )
        print(f"[Model] timm 骨干: {backbone_name}  |  pretrained={pretrained}")

    # ── 替换 Box Head（适配 num_classes）──
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # ── 替换 Mask Head（适配 num_classes）──
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes,
    )

    # ── 加载完整预训练权重（可选）──
    if mc.pretrained_weights is not None:
        state = torch.load(mc.pretrained_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state), strict=False)

    return model


def get_mask_rcnn_params(model: MaskRCNN, lr: float, weight_decay: float):
    """
    差异化学习率：
      backbone.*（含 FPN）使用 lr × 0.1，保护预训练特征
      rpn / roi_heads 使用标准 lr
    """
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": backbone_params, "lr": lr * 0.1, "weight_decay": weight_decay},
        {"params": other_params,    "lr": lr,        "weight_decay": weight_decay},
    ]
