"""
utils/logger.py
===============
统一日志管理：彩色控制台输出 + 文件持久化 + TensorBoard/WandB 写入器。

对外接口：
    logger = get_logger(name, log_dir)   # Python logger
    writer = get_tb_writer(log_dir)      # TensorBoard SummaryWriter
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 尝试导入彩色日志，不可用时退回标准格式
try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


# ──────────────────────────────────────────────────────────────────
# Python Logger
# ──────────────────────────────────────────────────────────────────

def get_logger(
    name: str = "iceberg",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """
    获取配置好的 Logger 实例。
    重复调用同一 name 时返回已有实例（避免重复 handler）。

    Args:
        name:    logger 名称（通常用模块名）
        log_dir: 若指定，则同时将日志写入 {log_dir}/train.log
        level:   日志级别，默认 INFO

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 已经初始化过则直接返回
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 不向 root logger 传播，避免重复输出

    # ── 控制台 Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if _HAS_COLORLOG:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s%(reset)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── 文件 Handler（可选）──
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")

    return logger


# ──────────────────────────────────────────────────────────────────
# TensorBoard / WandB Writer
# ──────────────────────────────────────────────────────────────────

def get_tb_writer(log_dir: str, run_name: Optional[str] = None):
    """
    创建 TensorBoard SummaryWriter。

    Args:
        log_dir:  TensorBoard 日志根目录
        run_name: 子目录名称（用于区分不同实验）。
                  若为 None 则自动使用时间戳。

    Returns:
        SummaryWriter 实例；若 tensorboard 未安装则返回 None。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[Logger] 警告: tensorboard 未安装，跳过 TensorBoard 初始化。")
        return None

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    tb_dir = Path(log_dir) / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[Logger] TensorBoard 写入目录: {tb_dir}")
    print(f"[Logger] 启动命令: tensorboard --logdir {Path(log_dir).resolve()}")
    return writer


def get_wandb_run(cfg, run_name: Optional[str] = None):
    """
    初始化 WandB 实验追踪。
    仅在 cfg.train.logging.use_wandb=True 时启用。

    Returns:
        wandb.Run 实例；不可用或未启用时返回 None。
    """
    if not cfg.train.logging.use_wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("[Logger] 警告: wandb 未安装，跳过 WandB 初始化。")
        return None

    run = wandb.init(
        project=cfg.train.logging.wandb_project,
        entity=cfg.train.logging.wandb_entity or None,
        name=run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        config=dict(cfg),   # 将完整配置记录到 WandB
        resume="allow",
    )
    return run
