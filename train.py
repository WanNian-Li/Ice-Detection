"""
train.py
=========
南极冰山检测与实例分割 —— 主训练脚本。

职责：串联逻辑，不含业务细节：
    加载配置 → 构建数据 → 构建模型 → 优化器/调度器
    → 训练循环（AMP + 梯度裁剪）→ 验证 → 断点保存 → 日志写入

使用方式：
    # 使用默认配置从头训练
    python train.py

    # 指定配置文件
    python train.py --config configs/config.yaml

    # 断点续训
    python train.py --resume outputs/checkpoints/last.pth

    # 覆盖单个配置项（OmegaConf 语法）
    python train.py train.epochs=100 train.optimizer.lr=5e-5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ── 将项目根目录加入路径 ──
sys.path.insert(0, str(Path(__file__).parent))

from configs.config_parser import get_config, save_config
from datasets.iceberg_dataset import build_dataloaders
from models.build import build_model, build_optimizer, build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import get_logger, get_tb_writer, get_wandb_run
from utils.losses import build_seg_loss
from utils.metrics import compute_instance_metrics, compute_semantic_iou


# ══════════════════════════════════════════════════════════════════
# 参数解析
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="冰山检测训练脚本")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="断点续训：指定 .pth 检查点路径",
    )
    # 支持命令行覆盖任意配置项，例如 train.epochs=50
    parser.add_argument("overrides", nargs="*", help="OmegaConf 配置覆盖项")
    # parse_known_args 兜底：overrides 与 --resume 顺序任意均可正确收集
    args, extra = parser.parse_known_args()
    args.overrides = args.overrides + extra
    return args


# ══════════════════════════════════════════════════════════════════
# 训练一个 Epoch（Mask R-CNN）
# ══════════════════════════════════════════════════════════════════

def train_one_epoch_maskrcnn(
    model, optimizer, loader, scaler, cfg, epoch, tb_writer, logger
) -> float:
    """
    Mask R-CNN 训练一个 epoch。
    模型在训练模式下接收 (images, targets)，直接返回 loss_dict。

    Returns:
        本 epoch 的平均总损失
    """
    model.train()
    device    = next(model.parameters()).device
    log_every = int(cfg.train.logging.log_every_n_steps)
    grad_norm = float(cfg.train.grad_clip.max_norm)
    use_amp   = cfg.train.amp and device.type == "cuda"

    total_loss = 0.0
    n_batches  = len(loader)
    pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", ncols=100, leave=False)

    for step, (images, targets) in enumerate(pbar):
        # ── 数据移到 GPU ──
        images  = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero_grad() 略快

        # ── 前向传播（AMP 混合精度）──
        with autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            # loss_dict 包含: loss_classifier, loss_box_reg,
            #                 loss_mask, loss_objectness, loss_rpn_box_reg
            losses = sum(loss for loss in loss_dict.values())

        # ── 反向传播 ──
        scaler.scale(losses).backward()

        # ── 梯度裁剪（防止爆炸）──
        if cfg.train.grad_clip.enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        scaler.step(optimizer)
        scaler.update()

        loss_val = losses.item()
        total_loss += loss_val

        # ── 进度条实时显示各分量损失 ──
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "mask": f"{loss_dict.get('loss_mask', torch.tensor(0)).item():.4f}",
            "rpn":  f"{loss_dict.get('loss_objectness', torch.tensor(0)).item():.4f}",
        })

        # ── TensorBoard 日志（每 log_every 步记录一次）──
        global_step = epoch * n_batches + step
        if tb_writer is not None and global_step % log_every == 0:
            tb_writer.add_scalar("train/total_loss", loss_val, global_step)
            for k, v in loss_dict.items():
                tb_writer.add_scalar(f"train/{k}", v.item(), global_step)

    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"[Train] Epoch {epoch:03d}  avg_loss={avg_loss:.4f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════════
# 训练一个 Epoch（U-Net）
# ══════════════════════════════════════════════════════════════════

def train_one_epoch_unet(
    model, optimizer, loader, criterion, scaler, cfg, epoch, tb_writer, logger
) -> float:
    """
    U-Net 语义分割训练一个 epoch。

    Returns:
        本 epoch 的平均总损失
    """
    model.train()
    device    = next(model.parameters()).device
    log_every = int(cfg.train.logging.log_every_n_steps)
    grad_norm = float(cfg.train.grad_clip.max_norm)
    use_amp   = cfg.train.amp and device.type == "cuda"

    total_loss = 0.0
    n_batches  = len(loader)
    pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", ncols=100, leave=False)

    for step, (images, masks) in enumerate(pbar):
        images = images.to(device)   # (B, 1, H, W)
        masks  = masks.to(device)    # (B, H, W) long

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(images)   # (B, C, H, W)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()

        if cfg.train.grad_clip.enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        global_step = epoch * n_batches + step
        if tb_writer is not None and global_step % log_every == 0:
            tb_writer.add_scalar("train/total_loss", loss_val, global_step)

    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"[Train] Epoch {epoch:03d}  avg_loss={avg_loss:.4f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════════
# 验证一个 Epoch
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, cfg, epoch, tb_writer, logger) -> dict:
    """
    在验证集上评估模型，返回指标字典。
    两种架构均支持；架构类型由 cfg.model.architecture 决定。

    Returns:
        metrics: dict，键值与 cfg.train.checkpoint.monitor_metric 对应
    """
    model.eval()
    device = next(model.parameters()).device
    arch   = cfg.model.architecture

    if arch == "mask_rcnn":
        metrics = _validate_maskrcnn(model, loader, cfg, device)
    else:
        metrics = _validate_unet(model, loader, cfg, device)

    # ── 打印 & TensorBoard ──
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"[Val]   Epoch {epoch:03d}  {metric_str}")

    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"val/{k}", v, epoch)

    return metrics


def _validate_maskrcnn(model, loader, cfg, device) -> dict:
    """Mask R-CNN 验证：收集所有预测和真值，统一计算实例级指标。"""
    all_preds   = []
    all_targets = []
    score_thresh = float(cfg.evaluate.score_threshold)

    for images, targets in tqdm(loader, desc="  Validating", ncols=80, leave=False):
        images = [img.to(device) for img in images]
        preds  = model(images)    # eval 模式直接返回预测结果

        # 将预测转回 CPU，masks 提前二值化：(N,1,H,W) float32 → (N,H,W) bool
        # 节省 4× 内存，防止大验证集 OOM（50GB float → 12GB bool）
        for p in preds:
            p_cpu = {k: v.cpu() for k, v in p.items()}
            if "masks" in p_cpu and p_cpu["masks"].numel() > 0:
                p_cpu["masks"] = (p_cpu["masks"].squeeze(1) > 0.5)
            all_preds.append(p_cpu)
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
        torch.cuda.empty_cache()

    return compute_instance_metrics(
        all_preds, all_targets,
        iou_thresh=float(cfg.evaluate.iou_thresholds[0]),
        score_thresh=score_thresh,
    )


def _validate_unet(model, loader, cfg, device) -> dict:
    """U-Net 二分类验证：sigmoid 阈值解码后计算语义分割 IoU。"""
    all_preds   = []
    all_targets = []

    for images, masks in tqdm(loader, desc="  Validating", ncols=80, leave=False):
        images = images.to(device)
        logits = model(images)                              # (B, 1, H, W)
        # sigmoid > 0.5 → binary LongTensor 0/1（与 compute_semantic_iou 兼容）
        preds  = (torch.sigmoid(logits.squeeze(1)) > 0.5).long().cpu()  # (B, H, W)
        all_preds.append(preds)
        all_targets.append(masks.long().cpu())              # float 0/1 → long 0/1

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # num_classes=2 仍然兼容（背景=0，冰山=1）
    return compute_semantic_iou(all_preds, all_targets, num_classes=2)


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 0. 解析参数 & 加载配置 ──────────────────────────────────
    args = parse_args()
    os.chdir(Path(__file__).parent)  # 保证相对路径从项目根目录解析

    cfg = get_config(args.config)

    # 命令行覆盖项（如 train.epochs=100）
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    arch = cfg.model.architecture

    # ── 1. 设备 ──────────────────────────────────────────────────
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", cfg.train.gpu_ids[0])
        torch.backends.cudnn.benchmark = True   # 固定输入尺寸时能加速
    else:
        device = torch.device("cpu")
        if cfg.train.device == "cuda":
            print("[警告] CUDA 不可用，已退回 CPU 训练。")

    # ── 2. 日志 & TensorBoard ──────────────────────────────────
    logger    = get_logger("iceberg", log_dir=cfg.paths.log_dir)
    tb_writer = get_tb_writer(cfg.paths.log_dir) if cfg.train.logging.use_tensorboard else None
    wandb_run = get_wandb_run(cfg)

    logger.info(f"设备: {device}  |  架构: {arch}  |  Epochs: {cfg.train.epochs}")

    # ── 3. 数据管道 ───────────────────────────────────────────
    logger.info("构建 DataLoader ...")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    if train_loader is None:
        logger.error(
            "训练集 DataLoader 为 None，请先运行 data_prep/prepare_dataset.py 生成数据集。"
        )
        return

    # ── 4. 模型 ──────────────────────────────────────────────
    logger.info("构建模型 ...")
    model = build_model(cfg).to(device)

    # ── 5. 优化器 & 调度器 ──────────────────────────────────
    optimizer = build_optimizer(cfg, model)
    scheduler, use_plateau = build_scheduler(cfg, optimizer)
    # use_plateau=True 时，scheduler 实际上是 (warmup, plateau) 元组

    # ── 6. AMP GradScaler ────────────────────────────────────
    scaler = GradScaler("cuda", enabled=(cfg.train.amp and device.type == "cuda"))

    # ── 7. U-Net 损失函数（Mask R-CNN 内部自带损失）────────────
    criterion = build_seg_loss(cfg).to(device) if arch == "unet" else None

    # ── 8. 断点续训 ──────────────────────────────────────────
    start_epoch = int(cfg.train.start_epoch)
    best_metric = 0.0 if cfg.train.checkpoint.mode == "max" else float("inf")

    if args.resume:
        _sched = scheduler[0] if use_plateau else scheduler
        start_epoch, best_metric = load_checkpoint(
            args.resume, model, optimizer, _sched, device=str(device)
        )
        logger.info(f"从 epoch {start_epoch} 继续训练，best_metric={best_metric:.4f}")

    # 保存本次实验配置（便于复现）
    save_config(cfg, Path(cfg.paths.log_dir) / "config_snapshot.yaml")

    # ── 9. 主训练循环 ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    monitor_metric   = cfg.train.checkpoint.monitor_metric
    is_higher_better = (cfg.train.checkpoint.mode == "max")

    # Early Stopping 计数器
    es_cfg      = cfg.train.early_stopping
    es_enabled  = bool(es_cfg.enabled)
    es_patience = int(es_cfg.patience)
    es_counter  = 0   # 连续未提升的 epoch 数

    _sched_for_ckpt = scheduler[0] if use_plateau else scheduler

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        epoch_start = time.time()

        # ── 训练 ──
        if arch == "mask_rcnn":
            train_loss = train_one_epoch_maskrcnn(
                model, optimizer, train_loader, scaler, cfg, epoch, tb_writer, logger
            )
        else:
            train_loss = train_one_epoch_unet(
                model, optimizer, train_loader, criterion, scaler,
                cfg, epoch, tb_writer, logger
            )

        # ── 验证 ──
        val_metrics = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, cfg, epoch, tb_writer, logger)
        else:
            logger.warning("验证集不存在，跳过验证。")

        # ── 学习率调度 ──
        if use_plateau:
            warmup_sched, plateau_sched = scheduler
            if epoch < int(cfg.train.lr_scheduler.warmup_epochs):
                warmup_sched.step()
            else:
                val_score = val_metrics.get(monitor_metric, train_loss)
                plateau_sched.step(val_score)
        else:
            scheduler.step()

        # ── 记录两个参数组的 lr（backbone × 0.1 vs FPN+Head）──
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head     = optimizer.param_groups[-1]["lr"]
        if tb_writer is not None:
            tb_writer.add_scalar("train/lr_backbone", lr_backbone, epoch)
            tb_writer.add_scalar("train/lr_head",     lr_head,     epoch)
            tb_writer.add_scalar("train/avg_loss",    train_loss,  epoch)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss,
                           "lr_backbone": lr_backbone, "lr_head": lr_head,
                           **val_metrics})

        # ── 判断是否刷新最佳 ──
        current_metric = val_metrics.get(monitor_metric, 0.0)
        if is_higher_better:
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = current_metric
            es_counter  = 0
            save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=_sched_for_ckpt,
                best_metric=best_metric,
            )
        else:
            es_counter += 1

        # ── 本 epoch 耗时 ──
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:03d}  "
            f"lr_backbone={lr_backbone:.2e}  lr_head={lr_head:.2e}  "
            f"best_{monitor_metric}={best_metric:.4f}  "
            f"es={es_counter}/{es_patience}  耗时={elapsed:.1f}s"
        )

        # ── Early Stopping 检查 ──
        if es_enabled and es_counter >= es_patience:
            logger.info(
                f"Early Stopping 触发：{es_patience} 个 epoch 内 "
                f"{monitor_metric} 无提升，训练终止。"
            )
            break

    # ── 10. 训练结束 ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"训练完成！最佳 {monitor_metric} = {best_metric:.4f}")
    logger.info(f"最佳模型: {cfg.paths.checkpoint_dir}/best_model.pth")
    logger.info("=" * 60)

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
