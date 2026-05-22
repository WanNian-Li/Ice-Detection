"""
models/build.py
================
模型工厂：根据 cfg.model.architecture 动态实例化对应网络。

使用方式：
    from models.build import build_model
    model = build_model(cfg)
"""

import torch.nn as nn

from models.mask_rcnn import build_mask_rcnn, get_mask_rcnn_params
from models.mask2former import build_mask2former, get_mask2former_params
from models.unet import build_unet
from models.yolo import build_yolo, get_yolo_params
from models.sam2_wrapper import build_sam2_model, get_sam2_params

# 支持的架构名称映射
_ARCH_REGISTRY = {
    "mask_rcnn":   build_mask_rcnn,
    "mask2former": build_mask2former,
    "unet":        build_unet,
    "yolo":        build_yolo,
    "sam2":        build_sam2_model,
}


def build_model(cfg) -> nn.Module:
    """
    根据 cfg.model.architecture 构建并返回模型实例。

    Args:
        cfg: OmegaConf 配置对象

    Returns:
        nn.Module（未移动到设备，由调用方负责 .to(device)）

    Raises:
        ValueError: 若 architecture 名称不在注册表中
    """
    arch = cfg.model.architecture
    if arch not in _ARCH_REGISTRY:
        raise ValueError(
            f"未知的模型架构: '{arch}'。"
            f"支持的架构: {list(_ARCH_REGISTRY.keys())}"
        )

    model = _ARCH_REGISTRY[arch](cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] 架构: {arch}  |  "
        f"可训练参数: {n_params / 1e6:.2f} M"
    )
    return model


def build_optimizer(cfg, model: nn.Module):
    """
    根据配置构建优化器。

    Mask R-CNN 自动使用差异化学习率（backbone × 0.1）；
    U-Net 所有层使用统一学习率。

    Args:
        cfg:   OmegaConf 配置对象
        model: 已实例化的模型

    Returns:
        torch.optim.Optimizer 实例
    """
    import torch.optim as optim

    opt_cfg = cfg.train.optimizer
    lr      = float(opt_cfg.lr)
    wd      = float(opt_cfg.weight_decay)

    # 差异化参数组（backbone × 0.1 vs head）
    if cfg.model.architecture == "mask_rcnn":
        param_groups = get_mask_rcnn_params(model, lr=lr, weight_decay=wd)
    elif cfg.model.architecture == "mask2former":
        param_groups = get_mask2former_params(model, lr=lr, weight_decay=wd)
    elif cfg.model.architecture == "yolo":
        param_groups = get_yolo_params(model, lr=lr, weight_decay=wd)
    elif cfg.model.architecture == "sam2":
        param_groups = get_sam2_params(model, lr=lr, weight_decay=wd)
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": wd}]

    opt_type = opt_cfg.type.lower()
    if opt_type == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=float(opt_cfg.momentum),
            nesterov=True,
        )
    elif opt_type == "adam":
        betas = tuple(opt_cfg.betas)
        optimizer = optim.Adam(param_groups, betas=betas)
    elif opt_type == "adamw":
        betas = tuple(opt_cfg.betas)
        optimizer = optim.AdamW(param_groups, betas=betas)
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")

    print(f"[Optimizer] 类型: {opt_type}  |  lr: {lr}  |  weight_decay: {wd}")
    return optimizer


def build_scheduler(cfg, optimizer):
    """
    构建带 Warmup 的学习率调度器。

    Warmup 阶段：前 warmup_epochs 个 epoch 线性增加 lr。
    Warmup 结束后：切换到 cfg.train.lr_scheduler.type 指定的调度策略。

    支持：
      - "cosine" : CosineAnnealingLR
      - "step"   : StepLR
      - "plateau": ReduceLROnPlateau（需在训练循环中传入验证指标）

    Returns:
        (scheduler, use_plateau)
        use_plateau=True 时，调用方须在验证后调用 scheduler.step(val_metric)
    """
    import torch.optim.lr_scheduler as sched

    tr_cfg    = cfg.train
    sch_cfg   = tr_cfg.lr_scheduler
    n_epochs  = int(tr_cfg.epochs)
    n_warmup  = int(sch_cfg.warmup_epochs)
    lr_init   = float(sch_cfg.warmup_lr_init)
    lr_base   = float(tr_cfg.optimizer.lr)

    # ── Warmup 阶段：LinearLR ──
    warmup = sched.LinearLR(
        optimizer,
        start_factor=max(lr_init / lr_base, 1e-8),
        end_factor=1.0,
        total_iters=n_warmup,
    )

    sch_type = sch_cfg.type.lower()
    use_plateau = False

    # ── 主调度器 ──
    if sch_type == "cosine":
        main = sched.CosineAnnealingLR(
            optimizer,
            T_max=max(n_epochs - n_warmup, 1),
            eta_min=lr_init,
        )
    elif sch_type == "cosine_restart":
        # 周期重启：T_0 为第一周期长度，T_mult 为后续倍增系数
        # 每次重启 LR 回到最高点，有效避免 cosine 跑满后 LR 永久低迷
        eta_min = float(getattr(sch_cfg, "eta_min", lr_init))
        main = sched.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sch_cfg.T_0),
            T_mult=int(sch_cfg.T_mult),
            eta_min=eta_min,
        )
    elif sch_type == "step":
        main = sched.StepLR(
            optimizer,
            step_size=int(sch_cfg.step_size),
            gamma=float(sch_cfg.gamma),
        )
    elif sch_type == "plateau":
        # ReduceLROnPlateau 不能直接放入 SequentialLR，单独返回
        main = sched.ReduceLROnPlateau(
            optimizer,
            mode=tr_cfg.checkpoint.mode,
            patience=int(sch_cfg.patience),
            factor=float(sch_cfg.factor),
            min_lr=lr_init,
        )
        use_plateau = True
    else:
        raise ValueError(f"不支持的调度器类型: {sch_type}")

    if use_plateau:
        # Warmup 阶段手动处理，主阶段直接返回 ReduceLROnPlateau
        print(
            f"[Scheduler] warmup({n_warmup} epochs) + ReduceLROnPlateau"
        )
        return (warmup, main), True

    # 将 warmup 和主调度器串联
    combined = sched.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[n_warmup],
    )
    print(f"[Scheduler] warmup({n_warmup} epochs) → {sch_type}")
    return combined, False
