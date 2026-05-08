"""
utils/losses.py
================
U-Net 语义分割损失函数。

Mask R-CNN 无需此模块（模型内部直接返回 loss_dict）。

提供：
    BinarySegLoss    — 二分类专用：BCEWithLogits + Binary Dice（★ 推荐）
    DiceLoss         — 多分类 Soft Dice（保留备用）
    FocalLoss        — 针对类别不均衡的 Focal Loss（保留备用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss，对类别极度不均衡（冰山像素稀少）的分割任务效果好。

    公式: Dice = 2*|P∩G| / (|P|+|G|+eps)
          Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) 未经 softmax 的 logits
            targets: (B, H, W) 类别索引，long 类型

        Returns:
            标量 Dice Loss（对所有前景类取均值）
        """
        num_classes = logits.shape[1]
        # one-hot 编码: (B, C, H, W)
        targets_oh = F.one_hot(targets, num_classes=num_classes)  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()       # (B,C,H,W)

        probs = torch.softmax(logits, dim=1)

        # 只对前景类（1 到 C-1）计算 Dice，背景类通常不纳入
        dice_per_class = []
        for c in range(1, num_classes):
            p = probs[:, c].contiguous().view(-1)
            g = targets_oh[:, c].contiguous().view(-1)
            intersection = (p * g).sum()
            dice = (2.0 * intersection + self.smooth) / (
                p.sum() + g.sum() + self.smooth
            )
            dice_per_class.append(1.0 - dice)

        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss，用于极度不均衡的二分类 / 多分类分割。
    alpha 控制类别权重，gamma 控制难样本聚焦程度。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) long
        """
        B, C, H, W = logits.shape
        # 重排为 (B*H*W, C)，计算 cross entropy
        logits_flat  = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        log_p = F.log_softmax(logits_flat, dim=1)
        p_t   = torch.exp(log_p.gather(1, targets_flat.unsqueeze(1))).squeeze(1)

        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = (-focal_weight * log_p.gather(
            1, targets_flat.unsqueeze(1)
        ).squeeze(1)).mean()
        return loss


class CombinedSegLoss(nn.Module):
    """
    推荐的 U-Net 分割损失：加权 CrossEntropy + Dice。

    Args:
        ce_weight:   CrossEntropy 项的权重（默认 0.5）
        dice_weight: Dice 项的权重（默认 0.5）
        class_weights: 各类别权重 Tensor，用于缓解类别不均衡
                        例如 [1.0, 10.0] 表示冰山类权重是背景的 10 倍
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) float32
            targets: (B, H, W)    long

        Returns:
            标量损失值
        """
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class BinarySegLoss(nn.Module):
    """
    二分类分割损失：BCEWithLogitsLoss（带 pos_weight）+ Binary Dice。

    优于多分类 CrossEntropy 的原因：
      - 2 通道 Softmax 对二分类是冗余参数，1 通道 Sigmoid 更简洁
      - pos_weight 直接放大冰山正样本梯度，比 class_weights 调参更直观
      - BCEWithLogitsLoss 数值稳定性优于手动 sigmoid + BCE

    Args:
        pos_weight: 冰山类（正样本）损失权重，对应 BCEWithLogitsLoss 的 pos_weight。
                    典型值 5–15，值越大对漏检的惩罚越重。
        smooth:     Dice 平滑项，防止分母为零。
    """

    def __init__(self, pos_weight: float = 10.0, smooth: float = 1.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W) 未经 sigmoid 的原始输出
            targets: (B, H, W)    float32，值域 {0.0, 1.0}

        Returns:
            标量损失：0.5 × BCE + 0.5 × Dice
        """
        targets_4d = targets.unsqueeze(1)   # (B, 1, H, W)

        # ── BCEWithLogits（数值稳定，内置 sigmoid）──
        pos_weight = torch.tensor(
            [self.pos_weight_val], dtype=logits.dtype, device=logits.device
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_4d, pos_weight=pos_weight
        )

        # ── Binary Dice ──
        probs = torch.sigmoid(logits)                   # (B, 1, H, W)
        p = probs.contiguous().view(-1)                 # (B*H*W,)
        g = targets_4d.contiguous().view(-1)            # (B*H*W,)
        inter = (p * g).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
            p.sum() + g.sum() + self.smooth
        )

        return 0.5 * bce_loss + 0.5 * dice_loss


def build_seg_loss(cfg) -> BinarySegLoss:
    """
    从配置文件构建 U-Net 二分类分割损失函数。
    冰山像素比例通常极低（<5%），pos_weight=10 放大正样本梯度补偿不均衡。
    """
    return BinarySegLoss(pos_weight=10.0, smooth=1.0)
