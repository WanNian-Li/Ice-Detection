"""
utils/checkpoint.py
====================
断点保存与恢复工具。

保存格式（.pth 字典）：
    {
        "epoch":          int,
        "model_state":    OrderedDict,
        "optimizer_state": dict,
        "scheduler_state": dict | None,
        "best_metric":    float,
        "cfg_yaml":       str,   # 序列化的配置（便于复现）
    }
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

logger = logging.getLogger("iceberg")


# ──────────────────────────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────────────────────────

def save_checkpoint(
    cfg,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_metric: float,
) -> Path:
    """
    将当前最佳模型覆盖写入 best_model.pth。
    只在验证指标刷新最佳时调用，不生成任何中间文件。

    Args:
        cfg:          配置对象
        epoch:        当前已完成的 epoch 编号（从 0 起）
        model:        模型（支持 DataParallel）
        optimizer:    优化器
        scheduler:    学习率调度器（可为 None）
        best_metric:  刷新后的最佳指标值

    Returns:
        保存路径 Path 对象（best_model.pth）
    """
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 兼容 DataParallel / DistributedDataParallel 包装
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    state = {
        "epoch":           epoch,
        "model_state":     model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric":     best_metric,
        "cfg_yaml":        OmegaConf.to_yaml(cfg),
    }

    best_path = ckpt_dir / "best_model.pth"
    torch.save(state, best_path)
    logger.info(
        f"[Checkpoint] 最佳模型已更新: {best_path}  "
        f"(epoch={epoch}  metric={best_metric:.4f})"
    )
    return best_path


def save_last_checkpoint(
    cfg,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_metric: float,
) -> Path:
    """每个 epoch 结束后覆盖写入 last_model.pth，用于断点续训。"""
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    state = {
        "epoch":           epoch,
        "model_state":     model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric":     best_metric,
        "cfg_yaml":        OmegaConf.to_yaml(cfg),
    }
    last_path = ckpt_dir / "last_model.pth"
    torch.save(state, last_path)
    return last_path


# ──────────────────────────────────────────────────────────────────
# 加载
# ──────────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> Tuple[int, float]:
    """
    从检查点文件中恢复模型、优化器、调度器状态。

    Args:
        checkpoint_path: .pth 文件路径
        model:           已实例化的模型（结构需与保存时一致）
        optimizer:       若为 None，则跳过优化器状态恢复
        scheduler:       若为 None，则跳过调度器状态恢复
        device:          加载目标设备

    Returns:
        (start_epoch, best_metric)
        start_epoch = 保存时的 epoch + 1（下一个待训练的 epoch）
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")

    logger.info(f"[Checkpoint] 加载: {path}")
    state = torch.load(path, map_location=device, weights_only=False)

    # ── 恢复模型参数 ──
    # 处理 DataParallel 保存时带 "module." 前缀的情况
    model_state = state["model_state"]
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state, strict=True)
    else:
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            # 尝试去除 "module." 前缀后重试
            stripped = {
                k.replace("module.", ""): v for k, v in model_state.items()
            }
            model.load_state_dict(stripped, strict=True)

    # ── 恢复优化器 ──
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    # ── 恢复调度器 ──
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    start_epoch  = state.get("epoch", 0) + 1
    best_metric  = state.get("best_metric", 0.0)

    logger.info(
        f"[Checkpoint] 恢复成功  "
        f"start_epoch={start_epoch}  best_metric={best_metric:.4f}"
    )
    return start_epoch, best_metric
