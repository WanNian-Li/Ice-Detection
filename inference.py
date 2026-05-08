"""
inference.py
=============
南极冰山大图滑窗推理脚本。

完整流程：
  1. 加载模型权重
  2. 读取 SAR GeoTIFF → 重投影到 EPSG:3031 → 归一化
  3. 滑窗切片推理（支持 Mask R-CNN 实例分割 / U-Net 语义分割）
  4. 跨窗口 NMS 去重（Mask R-CNN）或概率图融合（U-Net）
  5. 像素掩膜 → 地理多边形（保留仿射变换，输出真实地理坐标）
  6. 保存矢量文件（.gpkg / .shp / .geojson）
  7. 保存可视化结果图

使用方式：
    python inference.py --sar data/raw/sar/grid_1_hh_2018_10.tif
    python inference.py --sar /path/to/sar.tif --checkpoint outputs/checkpoints/best_model.pth
    python inference.py --sar /path/to/sar.tif --no_vis   # 跳过可视化
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import shape as shapely_shape, mapping, box
from torch.cuda.amp import autocast
from torchvision.ops import nms as torchvision_nms
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

sys.path.insert(0, str(Path(__file__).parent))
from configs.config_parser import get_config
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger

logger = get_logger("iceberg.inference")


# ══════════════════════════════════════════════════════════════════
# SAR 预处理（与 data_prep 保持完全一致）
# ══════════════════════════════════════════════════════════════════

def load_and_preprocess_sar(
    sar_path: Path,
    cfg,
) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    加载 SAR GeoTIFF，重投影到目标 CRS，归一化到 [0, 1]。

    Returns:
        (image_f32, transform, crs)
        image_f32: (H, W) float32，NaN 已替换为 0
    """
    target_crs = CRS.from_epsg(int(cfg.data_prep.target_crs.split(":")[1]))
    target_res = cfg.data_prep.get("target_resolution", None)
    band_idx   = int(cfg.data_prep.sar_bands[0])

    with rasterio.open(sar_path) as src:
        if target_res is not None:
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs, target_crs,
                src.width, src.height,
                *src.bounds,
                resolution=target_res,
            )
        else:
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs, target_crs,
                src.width, src.height,
                *src.bounds,
            )

        dst_array = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band_idx),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    # 归一化（与训练预处理完全相同，必须保持一致）
    norm_cfg = cfg.data_prep.normalize
    method   = norm_cfg.method

    if method == "db_clip":
        lo = float(norm_cfg.db_clip_min)
        hi = float(norm_cfg.db_clip_max)
    elif method == "percentile":
        p_low  = float(norm_cfg.percentile_low)
        p_high = float(norm_cfg.percentile_high)
        lo = float(np.nanpercentile(dst_array, p_low))
        hi = float(np.nanpercentile(dst_array, p_high))
    elif method == "minmax":
        lo = float(np.nanmin(dst_array))
        hi = float(np.nanmax(dst_array))
    else:
        raise ValueError(f"不支持的归一化方式: {method}，请检查 config.yaml")

    denom = hi - lo
    if denom < 1e-10:
        norm = np.zeros_like(dst_array, dtype=np.float32)
    else:
        norm = np.clip((dst_array - lo) / denom, 0.0, 1.0)

    norm = np.nan_to_num(norm, nan=0.0).astype(np.float32)

    logger.info(
        f"SAR 预处理完成: {dst_w}×{dst_h} px  "
        f"CRS=EPSG:{target_crs.to_epsg()}  "
        f"分辨率={target_res}m"
    )
    return norm, dst_transform, target_crs


# ══════════════════════════════════════════════════════════════════
# 滑窗推理引擎
# ══════════════════════════════════════════════════════════════════

class IcebergInferenceEngine:
    """
    支持 Mask R-CNN（实例分割）和 U-Net（语义分割）的大图滑窗推理。
    """

    def __init__(self, cfg, checkpoint_path: Optional[str] = None):
        self.cfg  = cfg
        self.arch = cfg.model.architecture

        # ── 设备 ──
        if cfg.train.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", cfg.train.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

        # ── 加载模型 ──
        # 推理时关闭 pretrained_backbone：checkpoint 已包含完整权重，
        # 无需从网络下载 ResNet 预训练权重再立刻覆盖（节省时间和流量）
        from omegaconf import OmegaConf
        infer_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"model": {"mask_rcnn": {"pretrained_backbone": False},
                                         "unet":      {"pretrained_encoder": False}}}),
        )
        self.model = build_model(infer_cfg).to(self.device)
        ckpt_path  = checkpoint_path or cfg.inference.checkpoint_path
        load_checkpoint(ckpt_path, self.model, device=str(self.device))
        self.model.eval()

        # ── 推理参数 ──
        sw = cfg.inference.sliding_window
        self.window_size  = int(sw.window_size)
        self.stride       = int(sw.stride)
        self.score_thresh = float(cfg.inference.score_threshold)
        self.nms_iou      = float(cfg.inference.nms_iou_threshold)
        self.use_amp      = (cfg.train.amp and self.device.type == "cuda")

        logger.info(
            f"推理引擎就绪  arch={self.arch}  device={self.device}  "
            f"window={self.window_size}  stride={self.stride}"
        )

    # ────────────────────────────────────────────────────────
    # 主入口
    # ────────────────────────────────────────────────────────

    def run(
        self,
        sar_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        对单张 SAR 大图执行完整推理流程。

        Args:
            sar_path:   SAR GeoTIFF 路径
            output_dir: 输出目录（为 None 时使用 cfg.paths.prediction_dir）
            visualize:  是否保存可视化结果图

        Returns:
            GeoDataFrame（各冰山实例的地理多边形 + 属性）
        """
        sar_path   = Path(sar_path)
        output_dir = Path(output_dir or self.cfg.paths.prediction_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = sar_path.stem

        # Step 1：加载并预处理 SAR
        image, transform, crs = load_and_preprocess_sar(sar_path, self.cfg)

        # Step 2：滑窗推理
        if self.arch == "mask_rcnn":
            gdf = self._infer_maskrcnn(image, transform, crs)
        else:
            gdf = self._infer_unet(image, transform, crs)

        logger.info(f"检测到 {len(gdf)} 个冰山实例")

        # Step 3：保存矢量
        if len(gdf) > 0:
            fmt      = self.cfg.inference.output_format
            vec_path = output_dir / f"{stem}_predictions.{fmt}"
            self._save_vector(gdf, vec_path, fmt)

        # Step 4：可视化
        if visualize and self.cfg.inference.save_visualization:
            vis_path = output_dir / f"{stem}_visualization.png"
            self._visualize(image, gdf, transform, vis_path)

        return gdf

    # ────────────────────────────────────────────────────────
    # Mask R-CNN 滑窗推理
    # ────────────────────────────────────────────────────────

    def _infer_maskrcnn(
        self,
        image: np.ndarray,
        transform: rasterio.Affine,
        crs: CRS,
    ) -> gpd.GeoDataFrame:
        """
        Mask R-CNN 滑窗推理。
        每个窗口的预测结果（boxes/scores/masks）转换为全图坐标后统一 NMS。
        """
        H, W = image.shape
        P    = self.window_size

        # 收集所有窗口的原始预测
        all_global_boxes  = []   # [x1, y1, x2, y2] in pixel (col,row)
        all_scores        = []
        all_local_masks   = []   # (N,1,H,W) float32，保持本地坐标
        all_offsets       = []   # (row_off, col_off)

        row_starts = self._get_window_starts(H, P, self.stride)
        col_starts = self._get_window_starts(W, P, self.stride)
        total_windows = len(row_starts) * len(col_starts)

        pbar = tqdm(total=total_windows, desc="  滑窗推理", ncols=80)
        for r in row_starts:
            for c in col_starts:
                pbar.update(1)

                patch = image[r: r + P, c: c + P]
                # 若切片不足 P×P（边界处），用 0 填充
                if patch.shape != (P, P):
                    pad_patch = np.zeros((P, P), dtype=np.float32)
                    ph, pw = patch.shape
                    pad_patch[:ph, :pw] = patch
                    patch = pad_patch

                # 构造模型输入：(3, P, P)，值域 [0,1]
                img_t = torch.from_numpy(
                    np.stack([patch] * 3, axis=0)
                ).unsqueeze(0).to(self.device)   # (1,3,P,P)

                with torch.no_grad(), autocast(enabled=self.use_amp):
                    preds = self.model([img_t[0]])[0]   # eval 模式返回 list

                # 过滤低置信度
                keep_mask = preds["scores"] >= self.score_thresh
                if keep_mask.sum() == 0:
                    continue

                scores  = preds["scores"][keep_mask].cpu()
                boxes   = preds["boxes"][keep_mask].cpu()   # (N,4) xyxy 本地坐标
                masks   = preds["masks"][keep_mask].cpu()   # (N,1,P,P)

                # 本地框 → 全图框（x/col 轴加 c，y/row 轴加 r）
                offset  = torch.tensor([c, r, c, r], dtype=torch.float32)
                global_boxes = boxes + offset

                all_global_boxes.append(global_boxes)
                all_scores.append(scores)
                all_local_masks.append(masks)
                all_offsets.append((r, c))

        pbar.close()

        if not all_global_boxes:
            logger.warning("未检测到任何冰山实例。")
            return gpd.GeoDataFrame(columns=["score", "area_km2", "geometry"],
                                    geometry="geometry", crs=crs)

        # 全图 NMS
        all_boxes_t  = torch.cat(all_global_boxes, dim=0)   # (M,4)
        all_scores_t = torch.cat(all_scores, dim=0)          # (M,)
        keep_idx     = torchvision_nms(all_boxes_t, all_scores_t, self.nms_iou)

        logger.info(
            f"NMS 前: {len(all_scores_t)} 个预测  "
            f"NMS 后: {len(keep_idx)} 个"
        )

        # 将保留的预测转换为地理多边形
        # 先把所有 local_mask + offset 展平成索引列表
        flat_masks   = []
        flat_offsets = []
        for masks_chunk, (r, c) in zip(all_local_masks, all_offsets):
            for i in range(len(masks_chunk)):
                flat_masks.append(masks_chunk[i])   # (1, P, P)
                flat_offsets.append((r, c))

        polygons  = []
        scores_out = []
        pixel_res  = abs(transform.a)   # 像素边长（米）

        for ki in keep_idx.tolist():
            mask_bin = (flat_masks[ki][0].numpy() > 0.5).astype(np.uint8)
            r_off, c_off = flat_offsets[ki]
            score = all_scores_t[ki].item()

            # 本地像素掩膜 → 地理多边形
            # 计算该窗口在全图仿射变换中的偏移变换
            window_transform = transform * rasterio.Affine.translation(c_off, r_off)
            polys = _mask_to_polygons(mask_bin, window_transform)

            for p in polys:
                area_km2 = p.area / 1e6   # m² → km²
                if area_km2 > 0:
                    polygons.append(p)
                    scores_out.append(score)

        if not polygons:
            return gpd.GeoDataFrame(columns=["score", "area_km2", "geometry"],
                                    geometry="geometry", crs=crs)

        areas_km2 = [p.area / 1e6 for p in polygons]
        gdf = gpd.GeoDataFrame(
            {"score": scores_out, "area_km2": areas_km2, "geometry": polygons},
            geometry="geometry",
            crs=crs,
        )
        return gdf

    # ────────────────────────────────────────────────────────
    # U-Net 滑窗推理
    # ────────────────────────────────────────────────────────

    def _infer_unet(
        self,
        image: np.ndarray,
        transform: rasterio.Affine,
        crs: CRS,
    ) -> gpd.GeoDataFrame:
        """
        U-Net 语义分割滑窗推理。
        使用概率图加权平均融合重叠区域，避免边界伪影。
        """
        import torch.nn.functional as F

        H, W  = image.shape
        P     = self.window_size
        # U-Net 输出 1 通道 sigmoid 概率（二分类）
        canvas = np.zeros((1, H, W), dtype=np.float32)
        count  = np.zeros((H, W),    dtype=np.float32)

        row_starts = self._get_window_starts(H, P, self.stride)
        col_starts = self._get_window_starts(W, P, self.stride)
        total_windows = len(row_starts) * len(col_starts)

        pbar = tqdm(total=total_windows, desc="  U-Net 推理", ncols=80)
        for r in row_starts:
            for c in col_starts:
                pbar.update(1)
                r_end = min(r + P, H)
                c_end = min(c + P, W)

                patch = image[r: r_end, c: c_end]
                if patch.shape != (P, P):
                    pad = np.zeros((P, P), dtype=np.float32)
                    pad[:patch.shape[0], :patch.shape[1]] = patch
                    patch = pad

                img_t = torch.from_numpy(patch[np.newaxis, np.newaxis]).to(self.device)

                with torch.no_grad(), autocast(enabled=self.use_amp):
                    logits = self.model(img_t)                        # (1, 1, P, P)
                    probs  = torch.sigmoid(logits)[0].cpu().numpy()   # (1, P, P)

                # 只累加有效区域（r_end-r × c_end-c），裁剪掉 padding 部分
                ph, pw = r_end - r, c_end - c
                canvas[:, r:r_end, c:c_end] += probs[:, :ph, :pw]
                count[r:r_end, c:c_end]      += 1

        pbar.close()

        # 归一化概率图（有效像素 count > 0）
        count_safe = np.maximum(count, 1)
        avg_prob   = (canvas[0] / count_safe)          # (H, W) sigmoid 平均概率
        score_thr  = float(self.cfg.inference.score_threshold)
        fg_mask    = (avg_prob >= score_thr).astype(np.uint8)  # (H, W) 二值图

        # 连通域分析：提取各冰山实例
        import cv2
        n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )

        polygons   = []
        scores_out = []
        pixel_res  = abs(transform.a)

        for lbl in range(1, n_labels):
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < int(self.cfg.data_prep.min_iceberg_pixels):
                continue

            inst_mask = (label_map == lbl).astype(np.uint8)
            # 前景类的平均置信度作为 score
            score_val = float(avg_probs[1][label_map == lbl].mean())

            polys = _mask_to_polygons(inst_mask, transform)
            for p in polys:
                if p.area > 0:
                    polygons.append(p)
                    scores_out.append(score_val)

        if not polygons:
            logger.warning("未检测到任何冰山实例。")
            return gpd.GeoDataFrame(columns=["score", "area_km2", "geometry"],
                                    geometry="geometry", crs=crs)

        areas_km2 = [p.area / 1e6 for p in polygons]
        gdf = gpd.GeoDataFrame(
            {"score": scores_out, "area_km2": areas_km2, "geometry": polygons},
            geometry="geometry",
            crs=crs,
        )
        return gdf

    # ────────────────────────────────────────────────────────
    # 通用工具
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _get_window_starts(total: int, window: int, stride: int) -> List[int]:
        """生成滑窗起始坐标列表，确保覆盖末尾区域。"""
        if total <= window:
            return [0]
        starts = list(range(0, total - window + 1, stride))
        if starts[-1] + window < total:
            starts.append(total - window)
        return starts

    def _save_vector(self, gdf: gpd.GeoDataFrame, path: Path, fmt: str):
        """保存矢量文件（gpkg / shp / geojson）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "gpkg":
            gdf.to_file(path, driver="GPKG")
        elif fmt == "shp":
            gdf.to_file(path, driver="ESRI Shapefile")
        elif fmt == "geojson":
            gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
        else:
            gdf.to_file(path, driver="GPKG")

        logger.info(f"矢量文件已保存: {path}  ({len(gdf)} 个要素)")

    def _visualize(
        self,
        image: np.ndarray,
        gdf: gpd.GeoDataFrame,
        transform: rasterio.Affine,
        output_path: Path,
        max_size: int = 2000,
    ):
        """
        生成两张可视化图：
          1. *_overview.png  — 全图概览（彩色填充多边形，无文字标注）
          2. *_zoom.png      — 置信度最高的前 30 个检测放大图（含分数标注）
        """
        import cv2

        H, W   = image.shape
        inv_t  = ~transform

        def _to_display(img_crop, crop_tf):
            """将裁剪区域等比缩放到 max_size 以内，返回 (disp_img, scale)。"""
            h, w = img_crop.shape
            if h > max_size or w > max_size:
                s = max_size / max(h, w)
                disp = cv2.resize(img_crop, (int(w * s), int(h * s)),
                                  interpolation=cv2.INTER_AREA)
            else:
                s    = 1.0
                disp = img_crop
            return disp, s

        def _draw_polygons(ax, gdf_sub, inv_crop, scale,
                           show_score=False, lw=1.5):
            """在 ax 上绘制 gdf_sub 中的多边形（像素坐标）。"""
            colors = plt.cm.tab20.colors
            for idx, (_, row) in enumerate(gdf_sub.iterrows()):
                geom  = row.geometry
                score = float(row.get("score", 1.0))
                color = colors[idx % len(colors)]
                coords = _geo_polygon_to_pixel(geom, inv_crop, scale)
                if coords is None:
                    continue
                for ring in coords:
                    xs, ys = zip(*ring)
                    ax.fill(xs, ys, alpha=0.40, color=color)
                    ax.plot(xs, ys, color=color, linewidth=lw)
                if show_score:
                    cx, cy = inv_crop * (geom.centroid.x, geom.centroid.y)
                    ax.text(cx * scale, cy * scale, f"{score:.2f}",
                            fontsize=7, color="white", ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="black", alpha=0.55, lw=0))

        # ════════════════════════════════════════
        # 图1：全图概览
        # ════════════════════════════════════════
        disp, scale = _to_display(image, transform)
        # 缩放自适应线宽：图像缩得越狠，线越粗，保证可见
        lw_overview = max(1.0, 2.5 / scale / max(H, W) * max_size)

        fig1, ax1 = plt.subplots(figsize=(14, int(14 * disp.shape[0] / disp.shape[1]) + 1))
        ax1.imshow(disp, cmap="gray", vmin=0, vmax=1)
        ax1.set_title(
            f"Iceberg Detection Overview  |  {len(gdf)} instances detected",
            fontsize=13, pad=8
        )
        ax1.axis("off")

        if len(gdf) > 0:
            _draw_polygons(ax1, gdf, inv_t, scale,
                           show_score=False, lw=lw_overview)

        overview_path = output_path.parent / (output_path.stem + "_overview.png")
        plt.tight_layout()
        plt.savefig(overview_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        logger.info(f"可视化（全图）已保存: {overview_path}")

        # ════════════════════════════════════════
        # 图2：最密集区域放大图（包含最多检测的固定窗口）
        # ════════════════════════════════════════
        if len(gdf) == 0:
            return

        top_n   = 30
        # 取所有检测的质心像素坐标
        all_cx  = np.array([inv_t * (r.geometry.centroid.x, r.geometry.centroid.y)
                             for _, r in gdf.iterrows()])   # shape (N, 2)  col, row
        # 滑动窗口大小（像素），在此范围内检测数最多的区域即为"最密集"
        win_px  = min(3000, max(H, W) // 2)

        # 统计每个检测在 win_px 邻域内的邻居数，选邻居最多的质心作为窗口中心
        best_idx, best_count = 0, -1
        for i, (cc, rr) in enumerate(all_cx):
            in_win = (
                (np.abs(all_cx[:, 0] - cc) < win_px / 2) &
                (np.abs(all_cx[:, 1] - rr) < win_px / 2)
            ).sum()
            if in_win > best_count:
                best_count, best_idx = in_win, i

        # 以最密集质心为中心，截取 win_px × win_px 的区域
        cc0, rr0 = all_cx[best_idx]
        r1 = max(0, int(rr0 - win_px / 2))
        c1 = max(0, int(cc0 - win_px / 2))
        r2 = min(H, r1 + win_px)
        c2 = min(W, c1 + win_px)
        # 确保窗口不因边界截断而变小
        r1 = max(0, r2 - win_px)
        c1 = max(0, c2 - win_px)

        # 只保留质心落在此窗口内的检测
        def _centroid_in_window(row):
            pc, pr = inv_t * (row.geometry.centroid.x, row.geometry.centroid.y)
            return c1 <= pc <= c2 and r1 <= pr <= r2

        gdf_zoom = gdf[[_centroid_in_window(row) for _, row in gdf.iterrows()]]
        # 同时也展示置信度最高的 top_n 个（取交集后再按分数排）
        if "score" in gdf_zoom.columns and len(gdf_zoom) > top_n:
            gdf_zoom = gdf_zoom.nlargest(top_n, "score")

        crop     = image[r1:r2, c1:c2]
        crop_tf  = transform * rasterio.Affine.translation(c1, r1)
        inv_crop = ~crop_tf

        disp_z, scale_z = _to_display(crop, crop_tf)
        lw_zoom  = max(1.5, 3.0 / max(scale_z, 0.05))

        fig2, ax2 = plt.subplots(figsize=(12, int(12 * disp_z.shape[0] / disp_z.shape[1]) + 1))
        ax2.imshow(disp_z, cmap="gray", vmin=0, vmax=1)
        ax2.set_title(
            f"Densest Detection Region  |  {len(gdf_zoom)} instances in {win_px}×{win_px}px window",
            fontsize=12, pad=8
        )
        ax2.axis("off")
        _draw_polygons(ax2, gdf_zoom, inv_crop, scale_z,
                       show_score=True, lw=lw_zoom)

        zoom_path = output_path.parent / (output_path.stem + "_zoom.png")
        plt.tight_layout()
        plt.savefig(zoom_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info(f"可视化（放大）已保存: {zoom_path}  ({len(gdf_zoom)} 个检测，窗口 {win_px}px)")


# ══════════════════════════════════════════════════════════════════
# 几何工具函数
# ══════════════════════════════════════════════════════════════════

def _mask_to_polygons(
    mask_uint8: np.ndarray,
    transform: rasterio.Affine,
    min_area_px: int = 4,
) -> list:
    """
    将二值掩膜（0/1 uint8）转换为带地理坐标的 Shapely Polygon 列表。
    使用 rasterio.features.shapes，自动应用仿射变换。

    Args:
        mask_uint8:  (H, W) uint8，1=前景
        transform:   像素坐标 → 地理坐标的仿射变换
        min_area_px: 面积小于此值的多边形将被过滤（去噪）

    Returns:
        Shapely Polygon 列表
    """
    polygons = []
    for geom_dict, val in rasterio_shapes(mask_uint8, transform=transform):
        if val != 1:
            continue
        poly = shapely_shape(geom_dict)
        if poly.is_valid and poly.area > 0:
            polygons.append(poly)
    return polygons


def _geo_polygon_to_pixel(geom, inv_transform, scale: float = 1.0):
    """
    将 Shapely 地理多边形的顶点转换为像素坐标（用于可视化绘制）。

    Returns:
        外环 + 内环的像素坐标列表，每个环为 [(x_px, y_px), ...]
        若几何为空则返回 None
    """
    from shapely.geometry import Polygon, MultiPolygon

    def _ring_to_pixel(ring_coords):
        pts = []
        for gx, gy in ring_coords:
            cx, cy = inv_transform * (gx, gy)
            pts.append((cx * scale, cy * scale))
        return pts

    rings = []
    if isinstance(geom, Polygon):
        rings.append(_ring_to_pixel(geom.exterior.coords))
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            rings.append(_ring_to_pixel(part.exterior.coords))
    else:
        return None

    return rings if rings else None


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="冰山 SAR 大图推理脚本")
    parser.add_argument(
        "--sar", "--input", dest="sar", type=str, required=True,
        help="输入 SAR GeoTIFF 路径",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型权重路径（为 None 时使用 config 中的 inference.checkpoint_path）",
    )
    parser.add_argument(
        "--output_dir", "--output", dest="output_dir", type=str, default=None,
        help="输出目录（为 None 时使用 config 中的 paths.prediction_dir）",
    )
    parser.add_argument(
        "--no_vis", action="store_true",
        help="跳过可视化（节省时间）",
    )
    parser.add_argument(
        "--score_thresh", type=float, default=None,
        help="置信度阈值（覆盖 config 中的 inference.score_threshold）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(Path(__file__).parent)

    cfg = get_config(args.config)

    # 命令行参数覆盖配置
    if args.score_thresh is not None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"inference": {"score_threshold": args.score_thresh}})
        )

    engine = IcebergInferenceEngine(cfg, checkpoint_path=args.checkpoint)
    gdf = engine.run(
        sar_path   = args.sar,
        output_dir = args.output_dir,
        visualize  = not args.no_vis,
    )

    if len(gdf) > 0:
        logger.info("预测结果统计：")
        logger.info(f"  总实例数:  {len(gdf)}")
        logger.info(f"  面积范围:  {gdf['area_km2'].min():.4f} – {gdf['area_km2'].max():.4f} km²")
        logger.info(f"  平均置信度: {gdf['score'].mean():.4f}")
    else:
        logger.info("未检测到冰山实例。")


if __name__ == "__main__":
    main()
