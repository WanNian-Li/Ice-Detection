"""
datasets/iceberg_dataset.py
============================
南极冰山数据集的 PyTorch Dataset 实现。

支持两种模型范式，由 cfg.model.architecture 控制：
  - "mask_rcnn"  : 返回 (image_tensor, target_dict)
                   target_dict 含 boxes / labels / masks / area / iscrowd
  - "unet"       : 返回 (image_tensor, mask_tensor)

数据增强通过 albumentations 实现，训练集启用、验证/测试集关闭。

使用方式：
    from datasets.iceberg_dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
"""

import os

# 禁用 albumentations 启动时的版本检查网络请求（离线环境必须）
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, RandomSampler

from utils.despeckle import get_despeckle_fn

# ──────────────────────────────────────────────────────────────────
# 数据增强流水线构建
# ──────────────────────────────────────────────────────────────────

def build_augmentation_pipeline(cfg, split: str) -> A.Compose:
    """
    根据配置和数据集分割类型，构建 albumentations 增强流水线。

    Args:
        cfg:   OmegaConf 配置对象
        split: "train" | "val" | "test"

    Returns:
        A.Compose 对象（albumentations 变换组合）
    """
    aug_cfg = cfg.dataset.augmentation
    img_size = int(cfg.dataset.image_size)

    # ── 仅训练集做几何增强 ──
    # SAR 数据策略说明：只做几何增强，不做任何像素级（强度）增强。
    #   1. 模糊（Blur）：SAR 冰山识别依赖斑点纹理的高频分量和边缘锐度，
    #      哪怕 3px 的模糊也会抹除这些诊断特征。
    #   2. 亮度/对比度：db_clip 归一化后 0/1 具有绝对物理含义（海面≈暗，冰山≈亮），
    #      随机偏移破坏冰山与海水的物理阈值界限。
    #   3. 几何增强（翻转/旋转/仿射）：冰山不具有方向性，是最安全有效的防过拟合手段。
    if split == "train" and aug_cfg.enabled:
        transforms = [
            # 几何变换（图像和实例 ID 掩膜同步；albumentations 对 mask 自动用 INTER_NEAREST）
            A.HorizontalFlip(p=float(aug_cfg.horizontal_flip_prob)),
            A.VerticalFlip(p=float(aug_cfg.vertical_flip_prob)),
            A.RandomRotate90(p=0.5),
            # albumentations 2.x 推荐用 Affine 替代 ShiftScaleRotate
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(1 - float(aug_cfg.scale_limit), 1 + float(aug_cfg.scale_limit)),
                rotate=(-int(aug_cfg.rotation_limit), int(aug_cfg.rotation_limit)),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ]

        # 可选：弹性形变（适合冰山形状多样化，但速度慢）
        if aug_cfg.elastic_transform:
            transforms.append(
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.3,
                )
            )

        # 最终：调整尺寸 + 转 Tensor
        transforms += [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ]

    else:
        # 验证 / 测试集：只做尺寸对齐，不做随机变换
        transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


# ──────────────────────────────────────────────────────────────────
# 核心 Dataset 类
# ──────────────────────────────────────────────────────────────────

class IcebergDataset(Dataset):
    """
    冰山数据集。

    __getitem__ 返回格式取决于 architecture：
      - "mask_rcnn": (image, target)
          image : FloatTensor (C, H, W)，值域 [0, 1]
          target: dict，包含 boxes / labels / masks / image_id / area / iscrowd
      - "unet"     : (image, mask)
          image : FloatTensor (C, H, W)，值域 [0, 1]
          mask  : LongTensor  (H, W)，值域 {0, 1, ...}（类别索引）
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        cfg,
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            csv_path:  划分 CSV 文件路径（由 prepare_dataset.py 生成）
            split:     "train" | "val" | "test"（用于日志和区分增强策略）
            cfg:       OmegaConf 配置对象
            transform: 若为 None，则自动根据 split 和 cfg 构建
        """
        self.split = split
        self.cfg = cfg
        self.architecture = cfg.model.architecture  # "mask_rcnn" | "unet"
        self.input_channels = int(cfg.dataset.input_channels)

        # ── 在线斑点降噪配置 ──
        self.despeckle_fn, self.despeckle_kwargs = get_despeckle_fn(cfg)
        self.despeckle_enabled = self.despeckle_fn is not None

        # ── 加载元数据 CSV ──
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"划分文件不存在: {csv_path}\n"
                f"请先运行 data_prep/prepare_dataset.py 生成数据集。"
            )
        df = pd.read_csv(csv_path)
        self.records = df.reset_index(drop=True)

        # ── 数据增强流水线 ──
        self.transform = transform or build_augmentation_pipeline(cfg, split)

    def __len__(self) -> int:
        return len(self.records)

    def _despeckle(self, image: np.ndarray) -> np.ndarray:
        """在线 SAR 斑点降噪（由配置控制启用/禁用与方法选择）。"""
        if not self.despeckle_enabled:
            return image
        return self.despeckle_fn(image, **self.despeckle_kwargs)

    def __getitem__(self, idx: int):
        """
        返回单个样本。
        """
        row = self.records.iloc[idx]

        image = np.load(row["image_path"])   # (H, W)，float32，[0, 1]
        mask  = np.load(row["mask_path"])    # (H, W)，uint16，值=冰山实例 ID

        # ── 在线 SAR 斑点降噪（仅对图像做，掩膜不变）──
        image = self._despeckle(image)

        # ── 通道扩展：albumentations 要求输入为 (H, W, C) ──
        # mask_rcnn / mask2former / yolo 均需要 3 通道输入；unet 保持单通道
        if self.architecture in ("mask_rcnn", "mask2former", "yolo"):
            image_hwc = np.stack([image] * 3, axis=-1)  # (H, W, 3)
        else:
            image_hwc = image[..., np.newaxis]           # (H, W, 1)

        # ── 执行数据增强 ──
        augmented = self.transform(image=image_hwc, mask=mask)
        image_tensor = augmented["image"].float()   # (C, H, W)，ToTensorV2 已转换
        mask_aug     = augmented["mask"]            # Tensor (H, W)

        # ── 按架构封装返回值 ──
        if self.architecture in ("mask_rcnn", "mask2former", "yolo"):
            return image_tensor, self._build_maskrcnn_target(mask_aug, idx)
        else:
            # U-Net 二分类：掩膜转为 float32 二值图
            return image_tensor, (mask_aug.int() > 0).float()

    # ────────────────────────────────────────
    # Mask R-CNN 目标字典构建
    # ────────────────────────────────────────

    def _build_maskrcnn_target(
        self, mask: torch.Tensor, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        从二值掩膜中提取 Mask R-CNN 所需的实例级目标字典。

        策略：对掩膜做连通域分析，每个连通域视为一个独立冰山实例。

        Args:
            mask: (H, W) 的 uint8 Tensor，0=背景，1=冰山
            idx:  样本在数据集中的全局索引

        Returns:
            dict:
              - boxes    : FloatTensor (N, 4)  [x1, y1, x2, y2]（XYXY 格式）
              - labels   : Int64Tensor (N,)    全为 1（冰山类）
              - masks    : BoolTensor  (N, H, W)
              - area     : FloatTensor (N,)    每个实例的掩膜面积（像素数）
              - iscrowd  : Int64Tensor (N,)    全为 0（非群体目标）
              - image_id : Int64Tensor (1,)
        """
        mask_np = mask.numpy()   # (H, W)，uint16，值 = 冰山实例 ID（0 = 背景）

        # ── 按实例 ID 提取各冰山，无需连通域分析 ──
        # .gpkg 中每个多边形已是独立实例，预处理阶段已烧录唯一 ID
        unique_ids = np.unique(mask_np)
        unique_ids = unique_ids[unique_ids > 0]   # 排除背景 ID=0

        boxes_list   = []
        labels_list  = []
        masks_list   = []
        areas_list   = []

        for iid in unique_ids:
            instance_mask = (mask_np == iid)   # (H, W)，bool
            area = int(instance_mask.sum())

            # 过滤面积极小的噪点（重投影或裁剪边缘产生的孤立像素）
            if area < 4:
                continue

            ys, xs = np.where(instance_mask)
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max() + 1), float(ys.max() + 1)

            # 确保 box 有正面积
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(1)                    # 冰山类别固定为 1
            masks_list.append(instance_mask)
            areas_list.append(float(area))

        # ── 构建无实例时的空目标（防止 collate_fn 报错）──
        if len(boxes_list) == 0:
            H, W = mask_np.shape
            return {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "masks":    torch.zeros((0, H, W), dtype=torch.bool),
                "area":     torch.zeros((0,),   dtype=torch.float32),
                "iscrowd":  torch.zeros((0,),   dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
            }

        return {
            "boxes":    torch.tensor(boxes_list,  dtype=torch.float32),
            "labels":   torch.tensor(labels_list, dtype=torch.int64),
            "masks":    torch.from_numpy(np.stack(masks_list, axis=0)),  # (N,H,W)
            "area":     torch.tensor(areas_list,  dtype=torch.float32),
            "iscrowd":  torch.zeros(len(boxes_list), dtype=torch.int64),
            "image_id": torch.tensor([idx],       dtype=torch.int64),
        }


# ──────────────────────────────────────────────────────────────────
# HDF5 Dataset：从打包好的 .h5 文件中高速读取数据
# ──────────────────────────────────────────────────────────────────

class IcebergHDF5Dataset(IcebergDataset):
    """
    从 pack_hdf5.py 生成的 HDF5 文件读取数据，其余逻辑与 IcebergDataset 完全相同。

    HDF5 文件结构（由 pack_hdf5.py 生成）：
        /images  float32 (N, H, W)   已归一化 SAR 图像
        /masks   uint16  (N, H, W)   冰山实例 ID 掩膜（0=背景）

    h5py 文件句柄采用"worker 内懒加载"策略：
        - __init__ 只读取长度，不保持文件打开
        - __getitem__ 首次调用时在当前 worker 进程内打开文件
        - 这样每个 DataLoader worker 拥有独立句柄，避免多进程共享 h5py 对象引发的死锁
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: str,
        cfg,
        transform: Optional[A.Compose] = None,
    ):
        # 不调用 IcebergDataset.__init__，直接初始化必要属性
        self.split = split
        self.cfg = cfg
        self.architecture = cfg.model.architecture
        self.input_channels = int(cfg.dataset.input_channels)

        # ── 在线斑点降噪配置 ──
        self.despeckle_fn, self.despeckle_kwargs = get_despeckle_fn(cfg)
        self.despeckle_enabled = self.despeckle_fn is not None

        self.h5_path = Path(h5_path)
        self.transform = transform or build_augmentation_pipeline(cfg, split)
        self._h5 = None   # 懒加载，每个 worker 独立打开

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 文件不存在: {self.h5_path}\n"
                f"请先运行 data_prep/pack_hdf5.py 生成 HDF5 数据集。"
            )

        # 只读一次以获取样本总数
        with h5py.File(str(self.h5_path), "r") as f:
            self._len = int(f["images"].shape[0])

    def __len__(self) -> int:
        return self._len

    def _get_h5(self) -> h5py.File:
        """在当前 worker 进程中懒加载 HDF5 文件句柄。"""
        if self._h5 is None:
            self._h5 = h5py.File(str(self.h5_path), "r")
        return self._h5

    def _copy_paste(
        self, f: h5py.File, dst_img: np.ndarray, dst_mask: np.ndarray
    ) -> tuple:
        """
        Copy-Paste 实例增强（Strategy A）。

        从随机样本（src）中抠取冰山实例，直接覆盖粘贴到当前样本（dst）的对应像素位置。
        SAR 图像中冰山亮度值具有绝对物理意义，不依赖上下文，直接像素替换语义合理。

        新实例分配不与 dst 冲突的 ID（dst_max_id + 1, +2, ...），
        _build_maskrcnn_target 按 ID 提取实例，无需额外处理。
        """
        aug_cfg = self.cfg.dataset.augmentation
        p = float(aug_cfg.get("copy_paste_prob", 0.0))
        if p <= 0.0 or np.random.random() > p:
            return dst_img, dst_mask

        src_idx  = np.random.randint(0, self._len)
        src_img  = f["images"][src_idx]
        src_mask = f["masks"][src_idx].astype(np.int32)

        src_ids = np.unique(src_mask)
        src_ids = src_ids[src_ids > 0]
        if len(src_ids) == 0:
            return dst_img, dst_mask

        # 随机选 1–3 个实例粘贴，避免过度覆盖原图内容
        n_paste  = np.random.randint(1, min(len(src_ids), 3) + 1)
        paste_ids = np.random.choice(src_ids, size=n_paste, replace=False)

        dst_max_id = int(dst_mask.max())
        new_img    = dst_img.copy()
        new_mask   = dst_mask.copy()

        for i, iid in enumerate(paste_ids):
            inst = (src_mask == iid)
            if inst.sum() < 4:
                continue
            new_img[inst]  = src_img[inst]
            new_mask[inst] = dst_max_id + i + 1

        return new_img, new_mask

    def __getitem__(self, idx: int):
        f = self._get_h5()
        image = f["images"][idx]              # (H, W)，float32，[0, 1]
        mask  = f["masks"][idx].astype(np.int32)  # uint16 → int32，PyTorch 不支持 uint16 比较

        # ── 在线 SAR 斑点降噪 ──
        image = self._despeckle(image)

        # ── Copy-Paste 增强（Strategy A，仅训练集，仅实例分割架构）──
        if self.split == "train" and self.architecture in ("mask_rcnn", "mask2former"):
            image, mask = self._copy_paste(f, image, mask)

        if self.architecture in ("mask_rcnn", "mask2former", "yolo"):
            image_hwc = np.stack([image] * 3, axis=-1)   # (H, W, 3)
        else:
            image_hwc = image[..., np.newaxis]            # (H, W, 1)

        augmented    = self.transform(image=image_hwc, mask=mask)
        image_tensor = augmented["image"].float()
        mask_aug     = augmented["mask"]

        if self.architecture in ("mask_rcnn", "mask2former", "yolo"):
            return image_tensor, self._build_maskrcnn_target(mask_aug, idx)
        else:
            return image_tensor, (mask_aug.int() > 0).float()


# ──────────────────────────────────────────────────────────────────
# collate_fn：Mask R-CNN 需要 list-of-dict 格式的 batch
# ──────────────────────────────────────────────────────────────────

def maskrcnn_collate_fn(batch):
    """
    Mask R-CNN DataLoader 的自定义 collate 函数。
    torchvision 的 Mask R-CNN 接受：
      images : List[Tensor]
      targets: List[Dict[str, Tensor]]
    而不是默认的堆叠 Tensor。
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ──────────────────────────────────────────────────────────────────
# 工厂函数：一键构建三个 DataLoader
# ──────────────────────────────────────────────────────────────────

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    根据配置，构建 train / val / test 三个 DataLoader。

    数据源优先级（当 dataloader.use_hdf5=true 时）：
        1. HDF5 文件（paths.hdf5_dir/{split}.h5）—— 存在则使用
        2. CSV + npy 文件（paths.split_dir/{split}.csv）—— 回退路径

    Args:
        cfg: OmegaConf 配置对象（由 get_config() 返回）

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dl_cfg = cfg.dataloader
    arch   = cfg.model.architecture

    hdf5_dir = Path(cfg.paths.hdf5_dir)

    # mask_rcnn / mask2former / yolo 均使用 list-of-dict 格式的 batch
    collate = maskrcnn_collate_fn if arch in ("mask_rcnn", "mask2former", "yolo") else None

    loaders = {}
    for split in ("train", "val", "test"):
        h5_path = hdf5_dir / f"{split}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 文件不存在: {h5_path}\n"
                f"请先运行 data_prep/pack_hdf5.py 生成数据集。"
            )
        dataset = IcebergHDF5Dataset(h5_path=h5_path, split=split, cfg=cfg)
        print(f"[DataLoader] {split:5s}: HDF5  {str(h5_path):<50s} ({len(dataset)} 个样本)")

        is_train = (split == "train")

        # ── 子集采样：训练集按 max_patches_per_epoch，验证集按 max_val_patches ──
        sampler = None
        shuffle = is_train and bool(dl_cfg.shuffle_train)
        if is_train:
            max_n = int(dl_cfg.get("max_patches_per_epoch") or 0)
        else:
            max_n = int(dl_cfg.get("max_val_patches") or 0) if split == "val" else 0
        if 0 < max_n < len(dataset):
            sampler = RandomSampler(dataset, replacement=False, num_samples=max_n)
            shuffle = False   # sampler 与 shuffle 互斥

        n_workers_train = int(dl_cfg.num_workers)
        # 验证/测试：减少 worker 数、关闭 persistent_workers，防止 /dev/shm 耗尽导致 SIGINT
        n_workers = n_workers_train if is_train else min(n_workers_train, 2)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(dl_cfg.batch_size),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=bool(dl_cfg.pin_memory),
            collate_fn=collate,
            drop_last=is_train,
            persistent_workers=(is_train and n_workers > 0),
            prefetch_factor=(4 if (is_train and n_workers > 0) else (2 if n_workers > 0 else None)),
        )

        n_per_epoch = max_n if (sampler is not None) else len(dataset)
        print(
            f"[DataLoader] {split:5s}: {len(dataset):6d} 个样本  "
            f"| 每epoch={n_per_epoch:6d}  "
            f"| batch_size={dl_cfg.batch_size}  "
            f"| workers={dl_cfg.num_workers}"
        )

    return loaders["train"], loaders["val"], loaders["test"]


# ──────────────────────────────────────────────────────────────────
# 快速自测：直接运行此文件验证 Dataset 能否正常工作
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.config_parser import get_config
    import os

    os.chdir(Path(__file__).parent.parent)
    cfg = get_config("configs/config.yaml")

    # ── 测试 Mask R-CNN 模式 ──
    print("\n=== 测试 Mask R-CNN 模式 ===")
    from omegaconf import OmegaConf
    cfg_mrcnn = OmegaConf.merge(cfg, OmegaConf.create({"model": {"architecture": "mask_rcnn"}}))
    train_loader, val_loader, test_loader = build_dataloaders(cfg_mrcnn)

    if train_loader is not None:
        images, targets = next(iter(train_loader))
        print(f"  Batch 图像数量: {len(images)}")
        print(f"  图像 shape: {images[0].shape}  dtype: {images[0].dtype}")
        print(f"  boxes shape: {targets[0]['boxes'].shape}")
        print(f"  masks shape: {targets[0]['masks'].shape}")
        print(f"  labels: {targets[0]['labels']}")
    else:
        print("  （CSV 文件不存在，跳过实际加载测试）")

    # ── 测试 U-Net 模式 ──
    print("\n=== 测试 U-Net 模式 ===")
    cfg_unet = OmegaConf.merge(cfg, OmegaConf.create({"model": {"architecture": "unet"}}))
    train_loader_u, _, _ = build_dataloaders(cfg_unet)

    if train_loader_u is not None:
        imgs, masks = next(iter(train_loader_u))
        print(f"  图像 shape: {imgs.shape}   dtype: {imgs.dtype}")
        print(f"  掩膜 shape: {masks.shape}  dtype: {masks.dtype}")
        print(f"  掩膜唯一值: {masks.unique().tolist()}")
    else:
        print("  （CSV 文件不存在，跳过实际加载测试）")

    print("\n自测完成。")
