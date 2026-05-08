"""
models/unet.py
===============
从零实现的标准 U-Net（Ronneberger et al., 2015）。

架构概览（以 input_channels=1, base_channels=64 为例）：
                      ┌────────────────────────────────┐
    Input (1,512,512) │  Encoder                       │
                      │  E1: DoubleConv → (64,512,512) │
                      │  E2: Pool → DoubleConv → (128,256,256) │
                      │  E3: Pool → DoubleConv → (256,128,128) │
                      │  E4: Pool → DoubleConv → (512,64,64)   │
                      │  Bottleneck: Pool → DoubleConv → (1024,32,32) │
                      │  Decoder                       │
                      │  D4: Up + cat(E4) → DoubleConv → (512,64,64)  │
                      │  D3: Up + cat(E3) → DoubleConv → (256,128,128)│
                      │  D2: Up + cat(E2) → DoubleConv → (128,256,256)│
                      │  D1: Up + cat(E1) → DoubleConv → (64,512,512) │
                      └────────────────────────────────┘
    Output (num_classes, 512, 512)

特点：
  - 支持任意输入通道数（SAR 单通道）
  - 上采样支持 bilinear（快）和 ConvTranspose2d（可学习）
  - BatchNorm + ReLU（标准配置）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# 基础模块
# ──────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    U-Net 基本卷积单元：Conv → BN → ReLU → Conv → BN → ReLU
    保持输入输出尺寸不变（padding=1）。
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """编码器下采样单元：MaxPool2d(2) → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    解码器上采样单元：上采样 → 拼接跳跃连接 → DoubleConv

    Args:
        bilinear: True 使用双线性插值（无参数，速度快）；
                  False 使用 ConvTranspose2d（可学习，略慢）
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            # 双线性上采样：先 ×2，再用 1×1 卷积减半通道数
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # 转置卷积：自动 ×2 并减半通道数
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    来自上一解码器层的特征图（低分辨率）
            skip: 来自对应编码器层的跳跃连接特征图（高分辨率）
        """
        x = self.up(x)

        # 处理尺寸不对齐（当输入尺寸为奇数时可能出现 1px 误差）
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        # 在通道维度拼接
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层：1×1 卷积将通道数映射为类别数"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────
# U-Net 主体
# ──────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    标准 U-Net。

    Args:
        in_channels:   输入通道数（SAR 单极化 = 1，双极化 = 2）
        num_classes:   输出类别数（二分类 = 2）
        base_channels: 第一个编码器层的输出通道数（默认 64）
        bilinear:      上采样方式（True=双线性，False=转置卷积）
    """

    def __init__(
        self,
        in_channels:   int  = 1,
        num_classes:   int  = 2,
        base_channels: int  = 64,
        bilinear:      bool = True,
    ):
        super().__init__()
        C = base_channels
        factor = 2 if bilinear else 1   # bilinear 时 UpBlock 用 in//2 中间层

        # ── 编码器 ──
        self.enc1 = DoubleConv(in_channels, C)          # → (C, H, W)
        self.enc2 = DownBlock(C,     C * 2)             # → (2C, H/2, W/2)
        self.enc3 = DownBlock(C * 2, C * 4)             # → (4C, H/4, W/4)
        self.enc4 = DownBlock(C * 4, C * 8)             # → (8C, H/8, W/8)

        # ── 瓶颈层 ──
        self.bottleneck = DownBlock(C * 8, C * 16 // factor)  # → (16C/f, H/16, W/16)

        # ── 解码器 ──
        self.dec4 = UpBlock(C * 16, C * 8  // factor, bilinear)
        self.dec3 = UpBlock(C * 8,  C * 4  // factor, bilinear)
        self.dec2 = UpBlock(C * 4,  C * 2  // factor, bilinear)
        self.dec1 = UpBlock(C * 2,  C,                bilinear)

        # ── 输出层 ──
        self.out_conv = OutConv(C, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """用 Kaiming 初始化卷积层，BN 初始化为单位映射。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)  输入图像（已归一化到 [0,1]）

        Returns:
            logits: (B, num_classes, H, W)  未经 softmax 的输出
        """
        # 编码器（保存跳跃连接）
        s1 = self.enc1(x)           # (B, C,   H,    W)
        s2 = self.enc2(s1)          # (B, 2C,  H/2,  W/2)
        s3 = self.enc3(s2)          # (B, 4C,  H/4,  W/4)
        s4 = self.enc4(s3)          # (B, 8C,  H/8,  W/8)
        bn = self.bottleneck(s4)    # (B, 16C, H/16, W/16)

        # 解码器（逐层上采样并拼接跳跃连接）
        d4 = self.dec4(bn, s4)      # (B, 8C,  H/8,  W/8)
        d3 = self.dec3(d4, s3)      # (B, 4C,  H/4,  W/4)
        d2 = self.dec2(d3, s2)      # (B, 2C,  H/2,  W/2)
        d1 = self.dec1(d2, s1)      # (B, C,   H,    W)

        return self.out_conv(d1)    # (B, num_classes, H, W)


def build_unet(cfg) -> UNet:
    """
    从配置文件构建 U-Net 实例。

    Args:
        cfg: OmegaConf 配置对象（读取 cfg.model.unet）

    Returns:
        UNet 实例
    """
    uc = cfg.model.unet
    return UNet(
        in_channels=int(uc.in_channels),
        num_classes=int(uc.num_classes),
        base_channels=64,
        bilinear=True,
    )
