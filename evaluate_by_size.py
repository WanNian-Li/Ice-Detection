"""
evaluate_by_size.py
===================
按冰山面积分层评估模型性能，诊断 precision 是被小冰山拖累还是全局性问题。

面积分层（基于论文的 A1-A5 分类，40m 分辨率下 1px = 0.0016 km²）：
  A1: 极小型   < 1 km²      (< 625 px)
  A2: 小型     1-10 km²     (625–6,250 px)
  A3: 中型     10-100 km²   (6,250–62,500 px)
  A4: 大型     > 100 km²    (> 62,500 px)

输出：
  1. 各尺寸分层的 GT / 预测 / TP / FP / FN 计数
  2. 各尺寸分层的 Precision / Recall / F1 / AP@0.50
  3. 假阳性的尺寸分布（判断小目标是否为 FP 主因）
  4. GT-Pred 交叉匹配矩阵（判断大小目标之间是否存在跨尺寸误匹配）

使用方式：
    python evaluate_by_size.py
    python evaluate_by_size.py --checkpoint outputs/checkpoints/best_model.pth --split val
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

logger = get_logger("iceberg.eval_by_size")

# ── 分辨率常数 ──
PIXEL_AREA_KM2 = 0.0016   # 40m × 40m = 1600 m² = 0.0016 km²

# ── 尺寸分层定义 ──
SIZE_BINS = [
    ("A1 (<1 km²)",      0.0,     1.0),
    ("A2 (1-10 km²)",    1.0,    10.0),
    ("A3 (10-100 km²)", 10.0,   100.0),
    ("A4 (>100 km²)",  100.0, float("inf")),
]


def px_to_km2(px_count) -> np.ndarray:
    return np.asarray(px_count) * PIXEL_AREA_KM2


def categorize_area_km2(area_km2: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= area_km2 < hi:
            return name
    return "A4 (>100 km²)"


# ══════════════════════════════════════════════════════════════════
# 分层评估主逻辑
# ══════════════════════════════════════════════════════════════════

def evaluate_stratified(
    all_predictions: List[Dict],
    all_targets:     List[Dict],
    iou_thresh:      float = 0.5,
    score_thresh:    float = 0.0,
) -> pd.DataFrame:
    """
    对预测结果按冰山面积分层计算指标。

    Returns:
        DataFrame，每行一个尺寸分层，列含 GT/Pred/TP/FP/FN/Precision/Recall/F1/AP/mIoU
    """
    # ── 初始化各层计数器 ──
    cat_names = [b[0] for b in SIZE_BINS]
    cats = {n: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0,
                 "ious": [], "detection_records": []}
            for n in cat_names}

    # size_gt 用于全局 PR 曲线
    global_detections = []   # (score, is_tp)
    total_gt_all = 0

    for pred, gt in zip(all_predictions, all_targets):
        scores     = pred["scores"].cpu().numpy()
        pred_masks = pred["masks"].cpu().numpy()
        gt_masks   = gt["masks"].cpu().numpy()

        # 过滤低置信度
        keep = scores >= score_thresh
        scores     = scores[keep]
        pred_masks = pred_masks[keep]

        n_pred = len(scores)
        n_gt   = len(gt_masks)
        total_gt_all += n_gt

        # ── 分类 GT 和预测的尺寸 ──
        gt_areas_px = np.array([int(m.sum()) for m in gt_masks.astype(bool)])
        gt_areas_km2 = px_to_km2(gt_areas_px)
        gt_cats = [categorize_area_km2(a) for a in gt_areas_km2]

        if n_pred > 0:
            if pred_masks.ndim == 4:
                pred_bin = (pred_masks[:, 0] > 0.5)
            else:
                pred_bin = pred_masks.astype(bool)
            pred_areas_px = np.array([int(m.sum()) for m in pred_bin])
        else:
            pred_bin = np.zeros((0, *gt_masks.shape[1:]), dtype=bool)
            pred_areas_px = np.array([], dtype=int)
        pred_areas_km2 = px_to_km2(pred_areas_px)
        pred_cats = [categorize_area_km2(a) for a in pred_areas_km2]

        # ── 统计各层 GT / Pred 数量 ──
        for c in gt_cats:
            cats[c]["gt"] += 1
        for c in pred_cats:
            cats[c]["pred"] += 1

        # ── 贪心匹配 (全局) ──
        if n_gt == 0:
            for i in range(n_pred):
                global_detections.append((float(scores[i]), False))
                cats[pred_cats[i]]["fp"] += 1
            continue

        if n_pred == 0:
            for c in gt_cats:
                cats[c]["fn"] += 1
            continue

        # 构建 IoU 矩阵
        gt_bin = gt_masks.astype(bool)
        iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i in range(n_pred):
            for j in range(n_gt):
                inter = (pred_bin[i] & gt_bin[j]).sum()
                union = (pred_bin[i] | gt_bin[j]).sum()
                iou_mat[i, j] = inter / (union + 1e-8)

        sort_idx = np.argsort(-scores)
        matched_gt = set()

        for si in sort_idx:
            best_j   = int(np.argmax(iou_mat[si]))
            best_iou = iou_mat[si, best_j]
            if best_iou >= iou_thresh and best_j not in matched_gt:
                global_detections.append((float(scores[si]), True))
                matched_gt.add(best_j)
                # 成功匹配：TP 计入 GT 和 Pred 对应层
                gt_cat = gt_cats[best_j]
                pr_cat = pred_cats[si]
                cats[gt_cat]["tp"] += 1
                cats[gt_cat]["ious"].append(float(best_iou))
            else:
                global_detections.append((float(scores[si]), False))
                cats[pred_cats[si]]["fp"] += 1

        # FN: 未匹配的 GT
        for j in range(n_gt):
            if j not in matched_gt:
                cats[gt_cats[j]]["fn"] += 1

    # ── 汇总各层指标 ──
    rows = []
    for name in cat_names:
        c = cats[name]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        avg_iou   = float(np.mean(c["ious"])) if c["ious"] else 0.0
        rows.append({
            "类别":        name,
            "GT 数量":     c["gt"],
            "Pred 数量":   c["pred"],
            "TP":         tp,
            "FP":         fp,
            "FN":         fn,
            "Precision":  round(precision, 4),
            "Recall":     round(recall, 4),
            "F1":         round(f1, 4),
            "Avg Mask IoU": round(avg_iou, 4),
        })

    # ── 全局行 ──
    global_tp = sum(c["tp"] for c in cats.values())
    global_fp = sum(c["fp"] for c in cats.values())
    global_fn = sum(c["fn"] for c in cats.values())
    global_prec = global_tp / (global_tp + global_fp + 1e-8)
    global_rec  = global_tp / (global_tp + global_fn + 1e-8)
    global_f1   = 2 * global_prec * global_rec / (global_prec + global_rec + 1e-8)
    all_ious = []
    for c in cats.values():
        all_ious.extend(c["ious"])
    global_iou = float(np.mean(all_ious)) if all_ious else 0.0

    # 全局 AP@0.50
    if global_detections:
        global_detections.sort(key=lambda x: -x[0])
        tp_cum = np.cumsum([int(r[1]) for r in global_detections])
        fp_cum = np.cumsum([int(not r[1]) for r in global_detections])
        prec_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
        rec_curve  = tp_cum / (total_gt_all + 1e-8)
        ap_global = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            prec_at_t = prec_curve[rec_curve >= t]
            ap_global += np.max(prec_at_t) if len(prec_at_t) > 0 else 0.0
        ap_global /= 101.0
    else:
        ap_global = 0.0

    rows.append({
        "类别":        "【全局合计】",
        "GT 数量":     total_gt_all,
        "Pred 数量":   sum(c["pred"] for c in cats.values()),
        "TP":         global_tp,
        "FP":         global_fp,
        "FN":         global_fn,
        "Precision":  round(global_prec, 4),
        "Recall":     round(global_rec, 4),
        "F1":         round(global_f1, 4),
        "Avg Mask IoU": round(global_iou, 4),
    })

    df = pd.DataFrame(rows)

    # ── 额外：FP 尺寸分布诊断 ──
    total_fp = global_fp
    if total_fp > 0:
        print("\n── 假阳性 (FP) 尺寸分布 ───────────────")
        for name in cat_names:
            fp_n = cats[name]["fp"]
            pct  = fp_n / total_fp * 100
            bar  = "█" * int(pct / 2)
            print(f"  {name:<20s}  FP={fp_n:5d}  ({pct:5.1f}%)  {bar}")

    return df


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    args = parse_args()
    os.chdir(Path(__file__).parent)

    cfg = get_config(args.config)
    arch = cfg.model.architecture

    if arch != "mask_rcnn":
        logger.error(f"此脚本仅支持 Mask R-CNN，当前架构: {arch}")
        return

    # ── 设备 ──
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", cfg.train.gpu_ids[0])
    else:
        device = torch.device("cpu")

    # ── 加载模型 ──
    logger.info(f"加载模型: {arch} ...")
    model = build_model(cfg).to(device)
    ckpt = args.checkpoint or cfg.inference.checkpoint_path
    load_checkpoint(ckpt, model, device=str(device))
    model.eval()

    # ── 数据集 ──
    split = args.split
    logger.info(f"加载 [{split}] 数据集 ...")
    train_l, val_l, test_l = build_dataloaders(cfg)
    loader_map = {"train": train_l, "val": val_l, "test": test_l}
    loader = loader_map.get(split)
    if loader is None:
        logger.error(f"[{split}] 数据集不存在或为空。")
        return

    use_amp = cfg.train.amp and device.type == "cuda"
    iou_thresh   = float(cfg.evaluate.iou_thresholds[0])
    score_thresh = float(cfg.evaluate.score_threshold)

    # ── 推理 ──
    all_preds   = []
    all_targets = []

    logger.info("开始推理 ...")
    for batch in tqdm(loader, desc=f"  {split}", ncols=80):
        images, targets = batch
        images_dev = [img.to(device) for img in images]

        with autocast(enabled=use_amp):
            preds = model(images_dev)

        all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    # ── 分层评估 ──
    logger.info("计算分层指标 ...")
    df = evaluate_stratified(
        all_preds, all_targets,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
    )

    # ── 打印表格 ──
    print("\n" + "=" * 85)
    print(f"  冰山检测分层评估 — split={split}  IoU_thresh={iou_thresh}  score_thresh={score_thresh}")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85)

    # ── 保存 CSV ──
    out_dir = Path(cfg.paths.prediction_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_by_size_{split}.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"分层评估结果已保存: {csv_path}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="冰山检测分层评估（按面积 A1-A4）"
    )
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="权重文件路径（为空则使用 config 中的路径）")
    parser.add_argument("--split",      type=str, default="val",
                        choices=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
