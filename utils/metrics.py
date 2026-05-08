"""
utils/metrics.py
=================
训练/验证阶段用到的评估指标。

── 语义分割（U-Net）──
    compute_semantic_iou(preds, targets, num_classes)  →  dict

── 实例分割（Mask R-CNN）──
    compute_instance_metrics(predictions, targets, iou_thresh)  →  dict

两个函数均返回形如 {"val_mask_iou": float, ...} 的字典，
与 config.yaml 中的 monitor_metric 键名对应。
"""

from typing import Dict, List

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────

def mask_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    计算两个二值掩膜之间的 IoU（Intersection over Union）。

    Args:
        pred_bin: (H, W) bool/uint8，预测掩膜
        gt_bin:   (H, W) bool/uint8，真值掩膜

    Returns:
        IoU ∈ [0, 1]；两者均为空时返回 1.0
    """
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0   # 预测和 GT 都是空 → 完全匹配
    return float(inter) / float(union)


def _greedy_match(
    pred_masks: np.ndarray,   # (N_pred, H, W) bool
    gt_masks:   np.ndarray,   # (N_gt,   H, W) bool
    iou_thresh: float,
) -> List[float]:
    """
    贪心匹配：对每个 GT 实例找 IoU 最高的预测，若 IoU >= 阈值则算命中。

    Returns:
        matched_ious: 每个命中的 (pred, gt) 对的 IoU 列表
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return []

    # 构建 IoU 矩阵 (N_pred, N_gt)
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_matrix[i, j] = mask_iou(pm, gm)

    matched_ious = []
    used_pred = set()
    used_gt   = set()

    # 按 IoU 从高到低依次匹配
    flat_indices = np.dstack(np.unravel_index(
        np.argsort(-iou_matrix.ravel()), iou_matrix.shape
    ))[0]

    for pi, gi in flat_indices:
        if pi in used_pred or gi in used_gt:
            continue
        iou = iou_matrix[pi, gi]
        if iou < iou_thresh:
            break
        matched_ious.append(iou)
        used_pred.add(pi)
        used_gt.add(gi)

    return matched_ious


# ──────────────────────────────────────────────────────────────────
# U-Net 语义分割指标
# ──────────────────────────────────────────────────────────────────

def compute_semantic_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    计算像素级语义分割的 IoU 指标。

    Args:
        preds:       (B, H, W) long，预测类别索引（argmax 后）
        targets:     (B, H, W) long，真值类别索引
        num_classes: 类别总数

    Returns:
        dict:
          "val_mask_iou"    : 前景类（冰山）IoU
          "val_miou"        : 所有类别的平均 IoU（背景 + 冰山）
          "val_dice"        : 前景类 Dice 系数
    """
    preds   = preds.cpu().numpy().astype(bool)    # (B, H, W)
    targets = targets.cpu().numpy().astype(bool)  # (B, H, W)

    per_class_iou = []
    for c in range(num_classes):
        pred_c = (preds   == c) if c > 0 else (~preds)     # 类别 c 的二值图
        gt_c   = (targets == c) if c > 0 else (~targets)

        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        iou   = float(inter) / float(union + 1e-8)
        per_class_iou.append(iou)

    # 前景类（冰山，class=1）的 IoU
    fg_iou = per_class_iou[1] if num_classes > 1 else per_class_iou[0]

    # Dice = 2 * |P∩G| / (|P| + |G|)
    pred_fg = preds.astype(np.uint8)
    gt_fg   = targets.astype(np.uint8)
    inter_d = (pred_fg & gt_fg).sum()
    dice    = float(2 * inter_d) / float(pred_fg.sum() + gt_fg.sum() + 1e-8)

    return {
        "val_mask_iou": fg_iou,
        "val_miou":     float(np.mean(per_class_iou)),
        "val_dice":     dice,
    }


# ──────────────────────────────────────────────────────────────────
# Mask R-CNN 实例分割指标
# ──────────────────────────────────────────────────────────────────

def compute_instance_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets:     List[Dict[str, torch.Tensor]],
    iou_thresh:  float = 0.5,
    score_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    计算 Mask R-CNN 的实例级指标（用于验证集监控）。

    Args:
        predictions: 模型 eval 模式输出的 List[Dict]，每个 dict 含：
                       - 'masks':  (N, 1, H, W) float32，概率图
                       - 'scores': (N,) float32
                       - 'labels': (N,) int64
        targets:     DataLoader 提供的 List[Dict]，每个 dict 含：
                       - 'masks':  (M, H, W) bool
                       - 'labels': (M,) int64
        iou_thresh:  匹配时使用的 IoU 阈值
        score_thresh: 预测置信度阈值（低于此值的预测被过滤）

    Returns:
        dict:
          "val_mask_iou" : 匹配实例的平均 Mask IoU（主要监控指标）
          "val_precision": 精确率 @ iou_thresh
          "val_recall"   : 召回率 @ iou_thresh
          "val_f1"       : F1 分数
    """
    all_ious  = []
    total_tp  = 0
    total_fp  = 0
    total_fn  = 0

    for pred, gt in zip(predictions, targets):
        # ── 过滤低置信度预测 ──
        scores = pred["scores"].cpu().numpy()
        keep   = scores >= score_thresh

        pred_masks_raw = pred["masks"][keep].cpu().numpy()     # (N,1,H,W) float 或 (N,H,W) bool
        gt_masks_raw   = gt["masks"].cpu().numpy()             # (M, H, W) bool

        # 兼容两种格式：validate 阶段已提前二值化为 (N,H,W) bool；
        # evaluate.py 传入的仍是原始 (N,1,H,W) float32，在此按需处理
        if len(pred_masks_raw) == 0:
            pred_masks = np.zeros((0, *gt_masks_raw.shape[1:]), dtype=bool)
        elif pred_masks_raw.ndim == 4:
            pred_masks = (pred_masks_raw[:, 0] > 0.5)
        else:
            pred_masks = pred_masks_raw.astype(bool)
        gt_masks = gt_masks_raw.astype(bool)

        n_pred = len(pred_masks)
        n_gt   = len(gt_masks)

        if n_gt == 0 and n_pred == 0:
            continue

        if n_gt == 0:
            total_fp += n_pred
            continue

        if n_pred == 0:
            total_fn += n_gt
            continue

        # 贪心匹配
        matched_ious = _greedy_match(pred_masks, gt_masks, iou_thresh)
        n_matched = len(matched_ious)

        all_ious.extend(matched_ious)
        total_tp += n_matched
        total_fp += n_pred - n_matched
        total_fn += n_gt   - n_matched

    # ── 汇总指标 ──
    mean_iou  = float(np.mean(all_ious)) if all_ious else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "val_mask_iou":  mean_iou,
        "val_precision": float(precision),
        "val_recall":    float(recall),
        "val_f1":        float(f1),
    }
