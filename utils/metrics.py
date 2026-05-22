"""
utils/metrics.py
=================
训练/验证阶段用到的评估指标。

── 语义分割（U-Net）──
    compute_semantic_iou(preds, targets, num_classes)  →  dict

── 实例分割（Mask R-CNN / YOLO）──
    InstanceMetricsAccumulator          流式累加器（逐图推入，只保留标量）
    compute_instance_metrics(...)  →  dict  批量接口（内部用累加器实现）

    主要指标：val_ap50（AP@IoU=0.5）
      - 对所有置信度阈值积分，不依赖固定阈值，是实例分割的标准 COCO 指标
    辅助指标：val_precision / val_recall / val_f1
      - 在固定 score_thresh 下计算，仅供参考，不用于模型保存决策
"""

from typing import Dict, List

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────

def mask_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """计算两个二值掩膜之间的 IoU。两者均为空时返回 1.0。"""
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO 101-point 插值 AP。recalls/precisions 已包含 (0, 1.0) 边界点。"""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 101.0


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
        dict: val_mask_iou / val_miou / val_dice
    """
    preds   = preds.cpu().numpy().astype(bool)
    targets = targets.cpu().numpy().astype(bool)

    per_class_iou = []
    for c in range(num_classes):
        pred_c = (preds   == c) if c > 0 else (~preds)
        gt_c   = (targets == c) if c > 0 else (~targets)
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        per_class_iou.append(float(inter) / float(union + 1e-8))

    fg_iou = per_class_iou[1] if num_classes > 1 else per_class_iou[0]

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
# 实例分割指标：流式累加器
# ──────────────────────────────────────────────────────────────────

class InstanceMetricsAccumulator:
    """
    流式实例分割指标累加器。

    用法：
        acc = InstanceMetricsAccumulator(iou_thresh=0.5, score_thresh=0.3)
        for pred, gt in zip(predictions, targets):
            acc.update(pred, gt)      # 逐图推入，掩膜即时释放
        metrics = acc.compute()

    内存开销：O(N_pred_total) 个标量，而非 O(N_pred × H × W) 个掩膜像素。
    对于 1000 张验证图、每张 20 个预测，仅需 ~160 KB（vs. 几十 GB 掩膜）。
    """

    def __init__(self, iou_thresh: float = 0.5, score_thresh: float = 0.3):
        self.iou_thresh  = iou_thresh
        self.score_thresh = score_thresh
        # AP50 需要：每个预测的置信度 + 是否 TP
        self._scores: List[float] = []
        self._is_tp:  List[bool]  = []
        self._n_gt:   int         = 0
        # F1 辅助计数器
        self._tp = self._fp = self._fn = 0

    def update(self, pred: dict, gt: dict) -> None:
        """
        推入单张图像的预测和真值，完成 IoU 匹配后立即丢弃掩膜张量。

        pred 须含:
            masks  (N, H, W) bool  或  (N, 1, H, W) float32
            scores (N,) float32
        gt 须含:
            masks  (M, H, W) bool
        """
        scores        = pred["scores"].cpu().numpy()           # (N,)
        pred_masks_raw = pred["masks"].cpu().numpy()           # (N,1,H,W) or (N,H,W)
        gt_masks      = gt["masks"].cpu().numpy().astype(bool) # (M, H, W)

        # 统一为 (N, H, W) bool
        if pred_masks_raw.ndim == 4:
            pred_masks = pred_masks_raw[:, 0] > 0.5
        else:
            pred_masks = pred_masks_raw.astype(bool)

        # GT mask 降采样与预测 mask 对齐（训练验证阶段已在 GPU 端 stride-4 降采样）
        # 避免在 CPU 上对 512×512 做大规模矩阵乘法；训练中趋势跟踪精度损失可忽略
        if pred_masks.shape[-1] > 0 and gt_masks.shape[-1] != pred_masks.shape[-1]:
            sh = gt_masks.shape[-2] // pred_masks.shape[-2]
            sw = gt_masks.shape[-1] // pred_masks.shape[-1]
            if sh > 1 or sw > 1:
                gt_masks = gt_masks[:, ::max(sh, 1), ::max(sw, 1)]

        n_pred = len(pred_masks)
        n_gt   = len(gt_masks)
        self._n_gt += n_gt

        # ── 无 GT：所有预测均为 FP ──
        if n_gt == 0:
            self._scores.extend(scores.tolist())
            self._is_tp.extend([False] * n_pred)
            self._fp += int((scores >= self.score_thresh).sum())
            return

        # ── 无预测：所有 GT 均为 FN ──
        if n_pred == 0:
            self._fn += n_gt
            return

        # ── 构建 IoU 矩阵 (n_pred, n_gt)：向量化版本 ──
        # 原双层 Python 循环是 O(n_pred × n_gt × H×W)，模型收敛后 n_pred 暴涨导致验证越来越慢
        # 矩阵乘法版本：inter = pred_flat @ gt_flat.T，由 BLAS 处理，速度提升 10-100×
        pred_flat  = pred_masks.reshape(n_pred, -1).astype(np.float32)  # (n_pred, H*W)
        gt_flat    = gt_masks.reshape(n_gt, -1).astype(np.float32)      # (n_gt, H*W)
        inter      = pred_flat @ gt_flat.T                               # (n_pred, n_gt)
        pred_area  = pred_flat.sum(1, keepdims=True)                    # (n_pred, 1)
        gt_area    = gt_flat.sum(1, keepdims=True).T                    # (1, n_gt)
        union      = pred_area + gt_area - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            iou_matrix = np.where(union > 0, inter / union, 1.0).astype(np.float32)

        # ── AP50 匹配：按置信度从高到低，每个 GT 至多匹配一次 ──
        score_order    = np.argsort(-scores)
        matched_gt_ap: set = set()
        for pi in score_order:
            row = iou_matrix[pi].copy()
            if matched_gt_ap:
                row[list(matched_gt_ap)] = -1.0
            best_gi = int(np.argmax(row))
            is_tp   = row[best_gi] >= self.iou_thresh
            if is_tp:
                matched_gt_ap.add(best_gi)
            self._scores.append(float(scores[pi]))
            self._is_tp.append(is_tp)

        # ── F1 匹配：过滤低置信度后按 IoU 贪心匹配 ──
        keep     = scores >= self.score_thresh
        n_kept   = int(keep.sum())
        if n_kept == 0:
            self._fn += n_gt
        else:
            kept_iou = iou_matrix[keep]
            matched_pred_f1: set = set()
            matched_gt_f1:  set = set()
            flat_order = np.dstack(np.unravel_index(
                np.argsort(-kept_iou.ravel()), kept_iou.shape
            ))[0]
            for pi, gi in flat_order:
                if pi in matched_pred_f1 or gi in matched_gt_f1:
                    continue
                if kept_iou[pi, gi] < self.iou_thresh:
                    break
                matched_pred_f1.add(pi)
                matched_gt_f1.add(gi)
            n_match   = len(matched_gt_f1)
            self._tp += n_match
            self._fp += n_kept - n_match
            self._fn += n_gt   - n_match

    def compute(self) -> Dict[str, float]:
        """汇总所有已推入图像，返回最终指标字典。"""
        # AP50
        if self._n_gt == 0 or not self._scores:
            ap50 = 0.0
        else:
            order      = np.argsort(-np.array(self._scores))
            tp_arr     = np.array(self._is_tp, dtype=np.float32)[order]
            tp_cumsum  = np.cumsum(tp_arr)
            fp_cumsum  = np.cumsum(1.0 - tp_arr)
            recalls    = tp_cumsum / self._n_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recalls    = np.concatenate([[0.0], recalls])
            precisions = np.concatenate([[1.0], precisions])
            ap50 = _compute_ap_101(recalls, precisions)

        # F1 辅助
        precision = self._tp / (self._tp + self._fp + 1e-8)
        recall    = self._tp / (self._tp + self._fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "val_ap50":      float(ap50),
            "val_precision": float(precision),
            "val_recall":    float(recall),
            "val_f1":        float(f1),
        }


# ──────────────────────────────────────────────────────────────────
# 批量接口（向后兼容，evaluate.py 仍可直接调用）
# ──────────────────────────────────────────────────────────────────

def compute_instance_metrics(
    predictions:  List[Dict[str, torch.Tensor]],
    targets:      List[Dict[str, torch.Tensor]],
    iou_thresh:   float = 0.5,
    score_thresh: float = 0.3,
) -> Dict[str, float]:
    """
    批量计算实例分割指标。内部使用 InstanceMetricsAccumulator，
    内存占用与流式版本一致（不会因数据集大小而暴涨）。
    """
    acc = InstanceMetricsAccumulator(iou_thresh=iou_thresh, score_thresh=score_thresh)
    for pred, gt in zip(predictions, targets):
        acc.update(pred, gt)
    return acc.compute()
