# 南极冰山检测与实例分割 — 项目说明

## 开发工作流

**本地（Windows）**：仅用于编写和修改代码，不在本地运行或调试。
**云端（Google Colab）**：所有训练、推理、调试均在云端执行，运行结果反馈后再回到本地修改代码。

因此：
- 不要建议在本地执行任何 `python` 命令
- 代码修改完成后，用户手动上传到云端服务器运行
- 报错信息和运行日志来自云端，粘贴到本地对话后再分析修复

## 环境（云端）

- 平台：Google Colab（数据存储在 Google Drive `/content/drive/MyDrive/`）
- Python：3.12
- PyTorch：2.5.1+cu121 | torchvision：0.20.1+cu121

---

## 项目结构

```
Iceberg-Detection/
├── configs/
│   ├── config.yaml          # 全局配置，所有超参数/路径在此集中管理
│   └── config_parser.py     # OmegaConf 加载 + 路径解析
├── data_prep/
│   └── prepare_dataset.py   # 预处理：SAR 重投影 → RTV → 实例掩膜 → 切片 → 划分
├── datasets/
│   └── iceberg_dataset.py   # PyTorch Dataset + DataLoader 工厂
├── models/
│   ├── mask_rcnn.py         # Mask R-CNN 封装（torchvision）
│   ├── unet.py              # 从零实现的 U-Net
│   └── build.py             # 模型/优化器/调度器工厂
├── utils/
│   ├── logger.py            # colorlog + TensorBoard + WandB
│   ├── checkpoint.py        # 断点保存/加载
│   ├── losses.py            # BinarySegLoss（BCEWithLogits + Dice）
│   ├── metrics.py           # compute_semantic_iou / compute_instance_metrics
│   └── despeckle.py         # SAR 斑点滤波 + RTV 平滑（见下文）
├── train.py                 # 主训练脚本
├── evaluate.py              # 测试集评估（AP50/AP75/mAP）
└── inference.py             # 大图滑窗推理 → GeoPackage 输出
```

---

## 数据来源与论文参考

- **数据集**：Chen et al. (2026, ESSD)《A six-year circum-Antarctic icebergs dataset (2018–2023)》
  - 数据：Sentinel-1 EW GRD HH 极化，每年 10 月，GEE 下载
  - 标注：.gpkg 矢量（每个多边形 = 一个冰山实例）
  - 覆盖 2018–2023 年，环南极全范围
- **参考代码**：论文附带 MATLAB pipeline（`Iceberg_detection_code/`），含完整的 TV平滑 + SLIC分割 + 增量RF分类步骤
  - **SLIC + 增量RF**：仅适用于论文的传统 ML 路线，**不引入** Mask R-CNN 路线
  - **RTV 平滑**：可直接迁移，已实现并集成到 `utils/despeckle.py`

---

## 数据流水线核心设计决策

### SAR 完整预处理顺序

```
GEE 下载（σ0 dB，COPERNICUS/S1_GRD 已含热噪声去除/辐射定标/地形校正）
  ↓
重投影 EPSG:4326 → EPSG:3031（南极极射赤面），40m 分辨率
  ↓
[可选] 离线 RTV 平滑（data_prep.tv_smooth.enabled=true）
  dB → 线性幅度（10^(x/20)，对应 MATLAB db2mag）→ RTV → dB
  ↓
切片（512×512，overlap=64px）
  ↓
dB 截断 [-30, +5] → 归一化 [0, 1]（db_clip，全局固定范围）
  ↓
保存 .npy（图像 float32，掩膜 uint16）
```

### SAR 预处理（prepare_dataset.py）

1. **坐标系**：SAR 从 EPSG:4326 重投影到 EPSG:3031（南极极射赤面），40m 分辨率
2. **归一化**：全局 dB 截断 [-30, +5] dB → [0, 1]（`method: "db_clip"`）
   - **禁止使用 minmax**：逐切片拉伸会破坏 dB 绝对物理意义，将海面噪声放大成假目标
3. **实例掩膜格式（重要）**：掩膜保存为 **uint16**，每个像素值 = 冰山实例 ID（非全为 1 的二值图）
   - .gpkg 中每个多边形特征 → 唯一整数 ID（1, 2, 3...）
   - 同一 MultiPolygon 的各部件保持相同 ID
   - 彻底解决密集区冰山粘连被错误合并的问题
4. **切片过滤**：NaN 比例 > 0.3 或前景像素 `(mask > 0).sum() < 10` 则丢弃
5. **数据集划分**：按 SAR 场景分组划分，防止空间数据泄露
6. **断点续跑**：`resume_enabled: true` 时自动跳过已完成场景

### Dataset（iceberg_dataset.py）

- **Mask R-CNN 模式**：
  - 图像扩展为 3 通道（单通道 SAR 复制 3 份，兼容 ResNet 骨干）
  - 从 uint16 实例 ID 掩膜中提取各实例（`np.unique` → 按 ID 提取，无需连通域分析）
  - 返回 `(List[Tensor], List[Dict[boxes/labels/masks/area/iscrowd/image_id]])`
- **U-Net 模式**：
  - 图像保持单通道
  - 掩膜转为 float32 二值图：`(mask > 0).float()`（用于 BCEWithLogitsLoss）
  - 返回 `(Tensor(1,H,W), Tensor(H,W) float)`
- **数据增强**：**仅几何增强**（flip/rotate/affine），不做任何像素级增强
  - 原因：SAR 斑点纹理是诊断特征；模糊会销毁边缘锐度；dB 归一化后 0/1 有绝对物理意义，亮度偏移破坏冰山/海水阈值
- `np.load(..., mmap_mode='c')` — copy-on-write，降低内存峰值

---

## RTV 平滑（utils/despeckle.py）

### 原理

**Relative Total Variation**（Xu et al., 2012 SIGGRAPH Asia `tsmooth.m`）：
通过分离图像的"结构"（冰山边缘）和"纹理"（SAR 斑点 + 海冰纹理），实现结构保留的平滑。

### 两种调用方式

| 函数 | 调用位置 | 输入/输出 | 用途 |
|------|---------|-----------|------|
| `rtv_smooth_db(img_db)` | `prepare_dataset.py`（离线） | dB → dB，保留 NaN | 精确对齐 MATLAB pipeline，推荐 |
| `despeckle_tv(image)` | `iceberg_dataset.py`（在线） | [0,1] → [0,1] | 不重跑预处理时的在线备选 |

### 关键参数（与 tsmooth.m 对齐）

```yaml
tv_smooth:           # 离线，data_prep 中
  enabled: false     # 改 true 以启用（需重新跑 prepare_dataset.py）
  weight: 0.01       # lambda = weight/2 = 0.005
  sigma: 3.0         # 初始纹理尺度，每迭代 ÷2，下限 0.5
  sharpness: 0.005   # epsilon_s 边缘灵敏度
  max_iter: 4        # 4 次迭代（与 tsmooth.m 一致）
```

### 与 MATLAB 的已知差异（可接受，无需修复）

1. **求解器预条件**：Python 使用无预条件 CG（tol=0.1）；MATLAB 使用 ichol-PCG。tol=0.1 对角占优矩阵收敛充分，结果差异可忽略。
2. **边界效应**：Python 512×512 patch 在 patch 边界可能有轻微拼接痕；MATLAB 用 2000×2000 块处理。64px overlap 已基本缓解，不影响训练。

### 切片尺寸决策

**保持 512×512，不改为 256×256**。理由：
- 模型 min/max_size=512 已对齐，无需改配置
- 物理覆盖 20.5km × 20.5km/patch，C/D 类冰山多落在单 patch 内
- 256×256 看似增加 4× patch 数，但信息量与场景数不变，不能解决数据不足问题
- MATLAB pipeline 使用 2000×2000 块，偏向大上下文

---

## 模型设计

### Mask R-CNN（主要方案）

- 骨干：ResNet-50 + FPN，ImageNet 预训练
- **差异化学习率**：骨干用 `lr × 0.1`，FPN + Head 用标准 `lr`
- **Anchor 尺寸（重要，已修正）**：`[[8],[16],[32],[64],[128]]`（5 级对应 P2-P6）
  - 原 `[[32],[64],[128],[256],[512]]` 存在严重问题：最小冰山约 5×5=25px，与 32×32 anchor 的 IoU ≈ 0.024，远低于 RPN 正样本阈值
  - 修正后：IoU(8×8, 5×5) ≈ 0.39，训练质量大幅提升
- 长宽比：`[0.5, 1.0, 2.0]`，覆盖细长到宽扁冰山
- 损失：模型内置（loss_classifier + loss_box_reg + loss_mask + loss_objectness + loss_rpn_box_reg）
- `box_detections_per_img: 300`（原 100，密集场景下旧值会硬截断 recall）
- `mask_roi_pool_output_size: 28`（原 14，提高小目标掩膜分辨率）
- 43.70M 可训练参数

### SAM2（提示式实例分割）

- 骨干：Hiera（hierarchical vision transformer），变体 tiny/small/base_plus/large
- **微调策略**：仅训练 `sam_prompt_encoder` + `sam_mask_decoder`（`freeze_image_encoder=true`）
  - 图像编码器 Hiera 已在 SA-1B 上预训练，SAR 特征可迁移；冻结可节省 ~60% 显存
  - 如需完整微调，设 `freeze_image_encoder: false`，图像编码器 lr × 0.01
- **训练提示**：GT 边界框作为提示词（box prompt）→ 预测对应实例掩膜
  - 调用 `predictor._prep_prompts` 将 box 转为 SAM2 内部角点格式（label=2/3）
  - 然后直接调用 `sam_prompt_encoder` + `sam_mask_decoder` 保留梯度链
- **推理**：`SAM2AutomaticMaskGenerator`（密集点网格提示），无需手工框
  - 按 `stability_score ≥ score_threshold` 过滤候选掩膜
  - `points_per_side=32` 对 512×512 图像生成 1024 个提示点
- **损失**：BCE（pos_weight=10）× 0.5 + Dice × 0.5 + IOU评分损失 × 0.05
- 差异化学习率：图像编码器 lr × 0.01，提示编码器 lr × 0.1，掩膜解码器标准 lr
- 参数量（tiny freeze_encoder）：约 4M 可训练参数

### U-Net（语义分割备选）

- 5 级编解码器，base_channels=64，双线性上采样
- **输出 1 通道**（sigmoid 二分类，非 2 通道 softmax）
  - 2 通道 softmax 对二分类冗余；BCEWithLogits + pos_weight 是遥感/医学分割标准实践
- Kaiming 初始化
- 17.26M 可训练参数

---

## 训练关键配置

- 优化器：AdamW，`lr=2e-4`，`weight_decay=1e-4`
- 调度器：LinearLR warmup（3 epochs）→ CosineAnnealingWarmRestarts（T_0=20，T_mult=2）
- AMP 混合精度：`GradScaler` + `autocast`，`unscale_` 在梯度裁剪前必须调用
- 梯度裁剪：`max_norm=1.0`
- 监控指标：`val_mask_iou`（越高越好）
- 损失函数（U-Net）：`BinarySegLoss = BCEWithLogits(pos_weight=10) × 0.5 + BinaryDice × 0.5`
- Early Stopping：`patience=25`（cosine_restart 周期末 LR 低迷，需更长窗口）

---

## 推理

- 滑窗推理：512×512 窗口，stride=448，重叠区域概率图加权平均
- Mask R-CNN：torchvision NMS 跨窗口去重
- U-Net：`sigmoid > score_threshold` 二值化 → 连通域分析提取实例
- 输出：GeoPackage (.gpkg) 矢量文件 + 两张可视化图（见下）
- **推理时强制覆盖 `pretrained_backbone: False`**，避免每次推理下载 ResNet50 权重（97.8MB）
- 参数别名：`--sar`/`--input` 均可指定输入文件；`--output_dir`/`--output` 均可指定输出目录

### 可视化输出（两个文件）

`*_visualization_overview.png`
- 全图缩放到 2000px 以内，scale-aware 线宽（始终约 2.5px 显示宽度）
- 彩色填充多边形，无文字标注，英文标题

`*_visualization_zoom.png`
- 置信度最高的前 30 个检测放大显示
- 含分数文字标注（fontsize=7，黑底白字）
- 裁剪至 top-30 实例的 bounding box 区域 + 5% padding

---

## 训练关键行为

- **只保存 `best_model.pth`**，不产生 epoch_XXX.pth / last.pth 等中间文件
- **Early Stopping**：`patience=25`，连续 25 个 epoch `val_mask_iou` 无提升则终止
- **日志同时记录两个参数组 LR**：`lr_backbone`（骨干）和 `lr_head`（FPN+Head）
- 每次训练开始自动保存 `outputs/logs/config_snapshot.yaml`（供复现验证）

---

## 运行方式（云端执行）

```python
# Colab：挂载 Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# 数据预处理
python data_prep/prepare_dataset.py --config configs/config.yaml

# 训练（从头）
python train.py

# 断点续训
python train.py --resume outputs/checkpoints/best_model.pth

# 命令行覆盖配置
python train.py train.epochs=100 train.optimizer.lr=5e-5

# 评估
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth

# 推理（--input 和 --sar 均可）
python inference.py --input data/raw/sar/new_scene.tif --output outputs/predictions/
```

---

## Colab 常见问题

### Google Drive 挂载断开（errno=107）

长时间预处理（数小时）时 Drive FUSE 挂载可能断开，表现为：
```
OSError: [Errno 107] Transport endpoint is not connected
```

代码已处理：检测到 errno=107 时打印提示并干净退出，**不会崩溃 traceback**。

处理方法：
```python
# 重新挂载
drive.mount('/content/drive', force_remount=True)
# 重新运行预处理（已完成场景自动跳过，需 resume_enabled: true）
```

### GDAL TIF 读取错误

```
TIFFFillTile:Read error ... got N bytes, expected M
```

文件在 GEE 下载时损坏或不完整。代码已自动跳过该场景（`err_no=1` 为警告级别）。修复方法：在 GEE 重新导出该 grid 文件。

---

## 已知问题 / 待办

- 当前数据量不足导致 `val_recall=0.38`，瓶颈是数据量而非模型设计；需补充覆盖冰山密集区的 SAR 场景后重新运行预处理。
- 新版可视化（overview + zoom 两文件）尚未在云端验证，下次推理后确认输出是否正常。
- `spatial_buffer_m: 0`（正式设置），测试时可临时调大以验证管道是否通畅。
- 离线 RTV（`tv_smooth.enabled: true`）尚未在全量数据上验证效果，仍设为 false；建议先在单个场景测试平滑效果再开启。
- `--resume` 只保留了 best_model.pth，断点续训仍可用但会从最佳 epoch 继续而非最后 epoch。
