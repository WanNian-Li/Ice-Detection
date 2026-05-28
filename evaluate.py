"""
evaluate.py
============
在测试集上进行全面模型评估，输出标准检测/分割指标。

指标体系：
  ── 实例分割（Mask R-CNN）──
    AP@IoU=0.50          (AP50，最常用)
    AP@IoU=0.75          (AP75，更严格)
    AP@IoU=0.50:0.95     (mAP，COCO 标准)
    Precision / Recall / F1  @ IoU=0.50
    平均 Mask IoU

  ── 语义分割（U-Net）──
    前景 IoU（冰山类）
    背景 IoU
    mIoU（平均类 IoU）
    Dice 系数
    像素准确率

使用方式：
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split test
    python evaluate.py --split val --no_vis
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).parent))

from configs.config_parser import get_config
from datasets.iceberg_dataset import build_dataloaders
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger

logger = get_logger("iceberg.evaluate")


# ══════════════════════════════════════════════════════════════════
# 后处理过滤：去除冰川 / 冰架误报
# ══════════════════════════════════════════════════════════════════

def filter_predictions_postprocess(
    preds:              List[Dict],
    max_area_px:        int   = 0,
    border_touch_area:  int   = 0,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    对推理结果进行后处理过滤，抑制冰川/冰架误报。

    两种互补启发式规则（无需外部地理数据）：
      1. 尺寸过滤 (max_area_px > 0)
             掩膜像素数 > max_area_px 的预测直接丢弃。
             依据：冰山有典型物理上限；冰架/冰川连片区域面积远超最大冰山。
             推荐初始值：50000 px（对应约 89km² @ 40m 分辨率）。
      2. 边缘连通过滤 (border_touch_area > 0)
             掩膜同时满足"触碰图像边缘"且"面积 > border_touch_area"时丢弃。
             依据：冰架通常从图像边缘延伸进 patch；孤立冰山即使碰边缘面积也小。
             推荐初始值：10000 px。

    Args:
        preds:             推理输出列表，每个元素含 masks(N,1,H,W) / scores(N,) / boxes(N,4)
        max_area_px:       尺寸过滤阈值（像素数），0 = 禁用
        border_touch_area: 边缘连通过滤面积下限（像素数），0 = 禁用

    Returns:
        (filtered_preds, stats)
        stats 包含 total_before / total_after / removed_area / removed_border
    """
    stats = {"total_before": 0, "total_after": 0, "removed_area": 0, "removed_border": 0}
    filtered = []

    for pred in preds:
        masks  = pred["masks"]   # (N, 1, H, W) float32
        scores = pred["scores"]  # (N,)
        boxes  = pred.get("boxes", torch.zeros(len(scores), 4))

        n = len(scores)
        stats["total_before"] += n

        if n == 0:
            filtered.append(pred)
            continue

        bin_masks = masks[:, 0] > 0.5          # (N, H, W) bool
        areas     = bin_masks.float().sum(dim=(-2, -1))  # (N,) 每个实例的掩膜面积

        keep = torch.ones(n, dtype=torch.bool)

        # ── 尺寸过滤 ──
        if max_area_px > 0:
            too_large = areas > max_area_px
            stats["removed_area"] += int(too_large.sum())
            keep &= ~too_large

        # ── 边缘连通过滤 ──
        if border_touch_area > 0:
            top    = bin_masks[:, 0, :].any(dim=1)
            bottom = bin_masks[:, -1, :].any(dim=1)
            left   = bin_masks[:, :, 0].any(dim=1)
            right  = bin_masks[:, :, -1].any(dim=1)
            touches = top | bottom | left | right
            border_large = touches & (areas > border_touch_area)
            stats["removed_border"] += int(border_large.sum())
            keep &= ~border_large

        idx = keep.nonzero(as_tuple=True)[0]
        stats["total_after"] += len(idx)
        filtered.append({
            "masks":  masks[idx],
            "scores": scores[idx],
            "boxes":  boxes[idx],
        })

    return filtered, stats


# ══════════════════════════════════════════════════════════════════
# AP 计算（COCO 101 点插值）
# ══════════════════════════════════════════════════════════════════

def compute_ap_from_pr(
    precisions: np.ndarray,
    recalls:    np.ndarray,
) -> float:
    """
    使用 101 点插值法计算 AP（Average Precision）。
    这是 COCO 评估协议中使用的标准方法。

    Args:
        precisions: 精确率数组，按召回率排序
        recalls:    召回率数组（0→1 升序）

    Returns:
        AP ∈ [0, 1]
    """
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        prec_at_t = precisions[recalls >= t]
        ap += np.max(prec_at_t) if len(prec_at_t) > 0 else 0.0
    return ap / 101.0


def compute_ap_at_iou(
    all_predictions: List[Dict],
    all_targets:     List[Dict],
    iou_thresh:      float,
    score_thresh:    float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    在指定 IoU 阈值下计算 AP、Precision、Recall、F1。

    Args:
        all_predictions: 每张图像的模型预测字典列表
                         每个字典含 "boxes", "scores", "masks"
        all_targets:     每张图像的真值字典列表
                         每个字典含 "boxes", "masks"
        iou_thresh:      匹配 IoU 阈值
        score_thresh:    预测置信度下限过滤

    Returns:
        (ap, precision, recall, f1)
    """
    # ── 收集所有预测 (score, is_tp) ──
    detection_records = []   # (score, is_tp)
    total_gt = 0

    for pred, gt in zip(all_predictions, all_targets):
        scores     = pred["scores"].cpu().numpy()
        pred_masks = pred["masks"].cpu().numpy()       # (N, 1, H, W) float32
        gt_masks   = gt["masks"].cpu().numpy()         # (M, H, W) bool/uint8

        # 过滤低置信度
        keep = scores >= score_thresh
        scores     = scores[keep]
        pred_masks = pred_masks[keep]

        n_pred = len(scores)
        n_gt   = len(gt_masks)
        total_gt += n_gt

        if n_gt == 0:
            for s in scores:
                detection_records.append((float(s), False))
            continue

        if n_pred == 0:
            continue

        # 二值化预测掩膜
        pred_bin = (pred_masks[:, 0] > 0.5)   # (N, H, W) bool
        gt_bin   = gt_masks.astype(bool)       # (M, H, W) bool

        # 构建 IoU 矩阵
        iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i in range(n_pred):
            for j in range(n_gt):
                inter = (pred_bin[i] & gt_bin[j]).sum()
                union = (pred_bin[i] | gt_bin[j]).sum()
                iou_mat[i, j] = inter / (union + 1e-8)

        # 按分数排序后贪心匹配
        sort_idx  = np.argsort(-scores)
        matched_gt = set()

        for si in sort_idx:
            best_j   = int(np.argmax(iou_mat[si]))
            best_iou = iou_mat[si, best_j]
            if best_iou >= iou_thresh and best_j not in matched_gt:
                detection_records.append((float(scores[si]), True))
                matched_gt.add(best_j)
            else:
                detection_records.append((float(scores[si]), False))

    if not detection_records or total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    # ── 按置信度排序，构建 PR 曲线 ──
    detection_records.sort(key=lambda x: -x[0])  # 降序
    tp_cum = np.cumsum([int(r[1]) for r in detection_records])
    fp_cum = np.cumsum([int(not r[1]) for r in detection_records])
    n_det  = len(detection_records)

    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall_curve    = tp_cum / (total_gt + 1e-8)

    ap = compute_ap_from_pr(precision_curve, recall_curve)

    # 固定阈值下的 P/R/F1（使用最优 F1 对应点）
    f1_curve = (2 * precision_curve * recall_curve /
                (precision_curve + recall_curve + 1e-8))
    best_idx  = int(np.argmax(f1_curve))
    precision = float(precision_curve[best_idx])
    recall    = float(recall_curve[best_idx])
    f1        = float(f1_curve[best_idx])

    return ap, precision, recall, f1


# ══════════════════════════════════════════════════════════════════
# 语义分割指标（U-Net）
# ══════════════════════════════════════════════════════════════════

def compute_unet_metrics(
    preds:       torch.Tensor,    # (N_total, H, W) long
    targets:     torch.Tensor,    # (N_total, H, W) long
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    计算语义分割全套指标。
    """
    preds_np   = preds.numpy()
    targets_np = targets.numpy()

    ious, dices = [], []
    for c in range(num_classes):
        pred_c = (preds_np == c)
        gt_c   = (targets_np == c)
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        ious.append(float(inter) / float(union + 1e-8))
        dices.append(float(2 * inter) / float(pred_c.sum() + gt_c.sum() + 1e-8))

    pixel_acc = float((preds_np == targets_np).mean())

    return {
        "iou_bg":       ious[0],
        "iou_iceberg":  ious[1],
        "miou":         float(np.mean(ious)),
        "dice_iceberg": dices[1],
        "pixel_acc":    pixel_acc,
        # 与 train.py 监控指标名保持一致
        "val_mask_iou": ious[1],
    }


# ══════════════════════════════════════════════════════════════════
# 可视化：预测样本展示网格
# ══════════════════════════════════════════════════════════════════

def visualize_predictions(
    images:      List[torch.Tensor],
    preds_list:  List,
    targets_list: List,
    arch:        str,
    output_path: Path,
    n_samples:   int = 8,
):
    """
    生成预测结果对比图（每行：SAR图 | 预测 | 真值）。

    Args:
        images:       图像张量列表，每个 (C, H, W)
        preds_list:   预测结果列表（格式取决于 arch）
        targets_list: 真值列表
        arch:         "mask_rcnn" | "unet"
        output_path:  保存路径
        n_samples:    展示样本数
    """
    n = min(n_samples, len(images))
    fig = plt.figure(figsize=(15, 5 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.4, wspace=0.05)

    for i in range(n):
        # SAR 图像（取第一通道，转为灰度）
        img_np = images[i][0].cpu().numpy()

        ax_img  = fig.add_subplot(gs[i, 0])
        ax_pred = fig.add_subplot(gs[i, 1])
        ax_gt   = fig.add_subplot(gs[i, 2])

        ax_img.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title("SAR 输入", fontsize=9)
        ax_img.axis("off")

        if arch == "mask_rcnn":
            pred  = preds_list[i]
            gt    = targets_list[i]

            # 预测掩膜叠加图
            pred_overlay = np.stack([img_np, img_np, img_np], axis=-1)
            masks_pred   = pred["masks"].cpu().numpy()   # (N, 1, H, W)
            for mi in range(len(masks_pred)):
                m = (masks_pred[mi, 0] > 0.5)
                pred_overlay[m, 0] = np.clip(pred_overlay[m, 0] + 0.4, 0, 1)  # 红色高亮

            # 真值掩膜叠加图
            gt_overlay = np.stack([img_np, img_np, img_np], axis=-1)
            masks_gt   = gt["masks"].cpu().numpy()       # (M, H, W)
            for mi in range(len(masks_gt)):
                m = masks_gt[mi].astype(bool)
                gt_overlay[m, 1] = np.clip(gt_overlay[m, 1] + 0.4, 0, 1)   # 绿色高亮

            score_info = (f"N_pred={len(masks_pred)}  "
                          f"score_avg={pred['scores'].mean().item():.2f}"
                          if len(pred['scores']) > 0 else "无预测")
            ax_pred.set_title(f"预测  {score_info}", fontsize=8)
            ax_gt.set_title(f"真值  N_gt={len(masks_gt)}", fontsize=9)
            ax_pred.imshow(pred_overlay)
            ax_gt.imshow(gt_overlay)

        else:   # U-Net
            pred_mask = preds_list[i].cpu().numpy()   # (H, W)
            gt_mask   = targets_list[i].cpu().numpy() # (H, W)
            ax_pred.imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
            ax_gt.imshow(gt_mask,   cmap="hot", vmin=0, vmax=1)
            ax_pred.set_title("预测掩膜", fontsize=9)
            ax_gt.set_title("真值掩膜", fontsize=9)

        ax_pred.axis("off")
        ax_gt.axis("off")

    plt.suptitle("评估样本可视化", fontsize=14, y=1.01)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"可视化已保存: {output_path}")


# ══════════════════════════════════════════════════════════════════
# 主评估函数
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    cfg,
    split:              str = "test",
    checkpoint_path:    Optional[str] = None,
    visualize:          bool = True,
    max_area_px:        int  = 0,
    border_touch_area:  int  = 0,
    compare:            bool = False,
) -> Dict[str, float]:
    """
    在指定分割数据集上全量评估模型。

    Args:
        max_area_px:       尺寸过滤阈值（像素数）。>0 时启用，推荐 50000。
        border_touch_area: 边缘连通过滤面积下限。>0 时启用，推荐 10000。
        compare:           True 时同时打印过滤前/后的指标对比表格。

    Returns:
        指标字典（过滤后），同时打印到控制台并保存为 CSV。
    """
    arch = cfg.model.architecture

    # ── 设备 ──
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", cfg.train.gpu_ids[0])
    else:
        device = torch.device("cpu")

    # ── 加载模型 ──
    logger.info(f"加载模型: {arch} ...")
    model = build_model(cfg).to(device)
    ckpt  = checkpoint_path or cfg.inference.checkpoint_path
    load_checkpoint(ckpt, model, device=str(device))
    model.eval()

    # ── 数据集 ──
    logger.info(f"加载 [{split}] 数据集 ...")
    train_l, val_l, test_l = build_dataloaders(cfg)
    loader_map = {"train": train_l, "val": val_l, "test": test_l}
    loader = loader_map.get(split)
    if loader is None:
        logger.error(f"[{split}] 数据集不存在或为空，请先运行预处理。")
        return {}

    num_classes = int(cfg.dataset.num_classes)
    use_amp     = cfg.train.amp and device.type == "cuda"

    # ── 推理 ──
    all_images   = []
    all_preds    = []
    all_targets  = []

    logger.info("开始推理 ...")
    for batch in tqdm(loader, desc=f"  {split}", ncols=80):
        if arch == "mask_rcnn":
            images, targets = batch
            images_dev = [img.to(device) for img in images]

            with autocast(enabled=use_amp):
                preds = model(images_dev)

            all_images.extend([img.cpu() for img in images])
            all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
            all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        else:  # U-Net
            images, masks = batch
            images_dev = images.to(device)

            with autocast(enabled=use_amp):
                logits = model(images_dev)   # (B, C, H, W)

            pred_masks = torch.argmax(logits, dim=1).cpu()  # (B, H, W)
            all_images.extend([images[i] for i in range(len(images))])
            all_preds.extend([pred_masks[i] for i in range(len(pred_masks))])
            all_targets.extend([masks[i] for i in range(len(masks))])

    # ── 后处理过滤（可选） ──
    filter_enabled = (max_area_px > 0 or border_touch_area > 0)
    preds_raw = all_preds   # 保留原始预测供对比

    if arch == "mask_rcnn" and filter_enabled:
        all_preds_filtered, fstats = filter_predictions_postprocess(
            all_preds,
            max_area_px=max_area_px,
            border_touch_area=border_touch_area,
        )
        removed = fstats["total_before"] - fstats["total_after"]
        logger.info(
            f"后处理过滤: 共 {fstats['total_before']} 个预测框 → "
            f"{fstats['total_after']} 个 "
            f"(移除 {removed}: "
            f"尺寸过大={fstats['removed_area']}, "
            f"边缘连通={fstats['removed_border']})"
        )
        all_preds = all_preds_filtered

    # ── 计算指标 ──
    logger.info("计算评估指标 ...")
    iou_thresholds = [float(t) for t in cfg.evaluate.iou_thresholds]
    score_thresh   = float(cfg.evaluate.score_threshold)

    if arch == "mask_rcnn":
        metrics = {}

        # AP 在各 IoU 阈值下
        ap_list = []
        for t in iou_thresholds:
            ap, prec, rec, f1 = compute_ap_at_iou(
                all_preds, all_targets,
                iou_thresh=t, score_thresh=score_thresh,
            )
            metrics[f"AP@{t:.2f}"] = ap
            metrics[f"P@{t:.2f}"]  = prec
            metrics[f"R@{t:.2f}"]  = rec
            metrics[f"F1@{t:.2f}"] = f1
            ap_list.append(ap)

        # mAP[0.5:0.95]
        ap_coco_list = []
        for t in np.arange(0.50, 1.00, 0.05):
            ap_t, _, _, _ = compute_ap_at_iou(
                all_preds, all_targets, iou_thresh=round(float(t), 2),
                score_thresh=score_thresh,
            )
            ap_coco_list.append(ap_t)
        metrics["mAP[0.50:0.95]"] = float(np.mean(ap_coco_list))

        # 平均 Mask IoU（所有匹配实例）
        from utils.metrics import compute_instance_metrics
        inst_m = compute_instance_metrics(
            all_preds, all_targets,
            iou_thresh=iou_thresholds[0],
            score_thresh=score_thresh,
        )
        metrics.update(inst_m)

    else:  # U-Net
        preds_cat   = torch.stack(all_preds,   dim=0)  # (N, H, W)
        targets_cat = torch.stack(all_targets, dim=0)  # (N, H, W)
        metrics = compute_unet_metrics(preds_cat, targets_cat, num_classes)

    # ── 打印结果 ──
    label = "过滤后" if filter_enabled else "原始"
    _print_metrics_table(metrics, split, arch, label=label)

    # ── compare 模式：额外计算并对比过滤前的指标 ──
    if compare and arch == "mask_rcnn" and filter_enabled:
        logger.info("── 计算过滤前（原始）指标以供对比 ──")
        metrics_raw: Dict[str, float] = {}
        for t in iou_thresholds:
            ap, prec, rec, f1 = compute_ap_at_iou(
                preds_raw, all_targets, iou_thresh=t, score_thresh=score_thresh,
            )
            metrics_raw[f"AP@{t:.2f}"] = ap
            metrics_raw[f"P@{t:.2f}"]  = prec
            metrics_raw[f"R@{t:.2f}"]  = rec
            metrics_raw[f"F1@{t:.2f}"] = f1

        ap_coco_raw = []
        for t in np.arange(0.50, 1.00, 0.05):
            ap_t, _, _, _ = compute_ap_at_iou(
                preds_raw, all_targets, iou_thresh=round(float(t), 2),
                score_thresh=score_thresh,
            )
            ap_coco_raw.append(ap_t)
        metrics_raw["mAP[0.50:0.95]"] = float(np.mean(ap_coco_raw))

        from utils.metrics import compute_instance_metrics as _cim
        metrics_raw.update(_cim(preds_raw, all_targets,
                                iou_thresh=iou_thresholds[0],
                                score_thresh=score_thresh))

        _print_compare_table(metrics_raw, metrics, split)

    # ── 保存 CSV ──
    out_dir  = Path(cfg.paths.prediction_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix   = "_filtered" if filter_enabled else ""
    csv_path = out_dir / f"eval_{split}_{arch}{suffix}.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"指标 CSV 已保存: {csv_path}")

    # ── 可视化样本 ──
    if visualize and len(all_images) > 0:
        vis_path = out_dir / f"eval_{split}_{arch}{suffix}_samples.png"
        visualize_predictions(
            all_images, all_preds, all_targets,
            arch=arch, output_path=vis_path, n_samples=8,
        )

    return metrics


def _print_metrics_table(
    metrics: Dict[str, float],
    split:   str,
    arch:    str,
    label:   str = "原始",
):
    """格式化打印指标表格。"""
    logger.info("=" * 55)
    logger.info(f"评估结果 [{label}]  split={split}  arch={arch}")
    logger.info("=" * 55)

    priority_keys = [
        "mAP[0.50:0.95]", "AP@0.50", "AP@0.75",
        "P@0.50", "R@0.50", "F1@0.50",
        "val_mask_iou", "val_precision", "val_recall", "val_f1",
        "miou", "iou_iceberg", "iou_bg", "dice_iceberg", "pixel_acc",
    ]
    shown = set()
    for k in priority_keys:
        if k in metrics:
            logger.info(f"  {k:<25s} {metrics[k]:.4f}")
            shown.add(k)
    for k, v in metrics.items():
        if k not in shown:
            logger.info(f"  {k:<25s} {v:.4f}")
    logger.info("=" * 55)


def _print_compare_table(
    raw:      Dict[str, float],
    filtered: Dict[str, float],
    split:    str,
):
    """并排打印过滤前/后的关键指标对比。"""
    keys = ["AP@0.50", "AP@0.75", "mAP[0.50:0.95]",
            "P@0.50", "R@0.50", "F1@0.50",
            "val_ap50", "val_precision", "val_recall", "val_f1"]
    keys = [k for k in keys if k in raw or k in filtered]

    logger.info("=" * 65)
    logger.info(f"过滤前后对比  split={split}")
    logger.info(f"  {'指标':<25s}  {'原始':>8s}  {'过滤后':>8s}  {'变化':>8s}")
    logger.info("-" * 65)
    for k in keys:
        v_raw = raw.get(k, float("nan"))
        v_fil = filtered.get(k, float("nan"))
        delta = v_fil - v_raw
        sign  = "+" if delta >= 0 else ""
        logger.info(f"  {k:<25s}  {v_raw:>8.4f}  {v_fil:>8.4f}  {sign}{delta:>7.4f}")
    logger.info("=" * 65)


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="冰山检测模型评估脚本")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="权重文件路径（为空则使用 config 中的路径）")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--no_vis",     action="store_true",
                        help="跳过可视化（节省内存）")

    # ── 后处理过滤（冰川/冰架抑制） ──
    filter_grp = parser.add_argument_group("后处理过滤")
    filter_grp.add_argument(
        "--max_area_px", type=int, default=0,
        help="尺寸过滤：掩膜像素数超过此值时丢弃（0=禁用）。推荐起始值：50000"
    )
    filter_grp.add_argument(
        "--border_touch_area", type=int, default=0,
        help="边缘连通过滤：触碰图像边缘且面积超过此值时丢弃（0=禁用）。推荐起始值：10000"
    )
    filter_grp.add_argument(
        "--compare", action="store_true",
        help="同时输出过滤前后的指标对比表格"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(Path(__file__).parent)

    cfg = get_config(args.config)
    evaluate(
        cfg,
        split=args.split,
        checkpoint_path=args.checkpoint,
        visualize=not args.no_vis,
        max_area_px=args.max_area_px,
        border_touch_area=args.border_touch_area,
        compare=args.compare,
    )


if __name__ == "__main__":
    main()
