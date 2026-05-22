"""
data_prep/prepare_dataset.py
============================
南极冰山数据集预处理主脚本。

完整流程：
  1. 扫描 SAR GeoTIFF 影像，按年月匹配对应 .gpkg 矢量文件
  2. 将 SAR 影像从 EPSG:4326 重投影到目标坐标系（EPSG:3031）
  3. 将矢量多边形重投影并烧录为像素级二值掩膜（rasterize）
  4. 滑窗切图，提取 patch_size × patch_size 大小的图像块和掩膜块
  5. 过滤无效切片（NaN 比例过高 / 冰山像素不足）
  6. 归一化 SAR 强度，保存为 .npy 格式
  7. 生成 train / val / test 划分列表 CSV

使用方式：
    python data_prep/prepare_dataset.py --config configs/config.yaml
    python data_prep/prepare_dataset.py --config configs/config.yaml --dry_run
"""

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import (GeometryCollection, MultiPolygon, Polygon,
                               box, mapping)
from tqdm import tqdm

# 将项目根目录加入路径，以便在任意位置运行此脚本
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config
from utils.despeckle import rtv_smooth_db

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────

def extract_year_month(filepath: Path) -> Optional[str]:
    """
    从文件名中提取年月字符串，例如：
      'grid_1_hh_2018_10.tif' → '201810'
      'sar_2020_10_hv.tif'    → '202010'
    若无法解析则返回 None。
    """
    stem = filepath.stem
    # 匹配形如 YYYY_MM 的年月模式
    match = re.search(r'(20\d{2})_?(0[1-9]|1[0-2])', stem)
    if match:
        year = match.group(1)
        month = match.group(2)
        return f"{year}{month}"
    return None


def flatten_geometry(geom):
    """
    将 GeometryCollection 展平为多边形列表。
    只保留 Polygon 和 MultiPolygon，丢弃 LineString / Point 等噪声几何。
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.is_valid and not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon)
                and g.is_valid and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            result.extend(flatten_geometry(g))
        return result
    return []


def normalize_sar(
    patch:      np.ndarray,
    method:     str   = "db_clip",
    db_min:     float = -30.0,
    db_max:     float =   5.0,
    p_low:      float =   2.0,
    p_high:     float =  98.0,
) -> np.ndarray:
    """
    对单张 SAR 切片进行强度归一化，输出范围 [0, 1]（float32）。
    NaN 值归一化后替换为 0（表示无数据区域）。

    Args:
        patch:   (H, W) float，SAR dB 值
        method:  归一化方式（见下方说明）
        db_min:  'db_clip' 方式的截断下限（dB），默认 -30.0
        db_max:  'db_clip' 方式的截断上限（dB），默认 +5.0
        p_low:   'percentile' 方式的低百分位
        p_high:  'percentile' 方式的高百分位

    归一化方式说明：
        'db_clip'   ★ 推荐：全局截断到 [db_min, db_max] dB 后线性映射 [0,1]。
                    保留 dB 值的绝对物理含义（海面≈暗，冰山≈亮），
                    避免逐切片拉伸将纯海面噪声误放大为高对比度假目标。
                    依据实测数据：p01=-27dB, p99=-0.4dB, max=+6.3dB。

        'percentile' 全局百分位截断（适合批量统计后固定 lo/hi 再使用）。

        'minmax'    逐切片局部拉伸（不推荐用于 SAR dB 数据：会破坏跨切片
                    的强度一致性，将纯海面噪声放大至与冰山相似的对比度）。

        'zscore'    标准化为均值0/方差1，适合强度动态范围很大的非dB数据。

    Returns:
        归一化后的 float32 数组，形状 (H, W)
    """
    valid = patch[~np.isnan(patch)]
    if valid.size == 0:
        return np.zeros_like(patch, dtype=np.float32)

    if method == "db_clip":
        # 全局截断：消除 -131dB 等极端异常值，保留物理意义
        lo, hi = db_min, db_max

    elif method == "minmax":
        lo, hi = np.nanmin(patch), np.nanmax(patch)

    elif method == "percentile":
        lo = np.nanpercentile(patch, p_low)
        hi = np.nanpercentile(patch, p_high)

    elif method == "zscore":
        mean = np.nanmean(patch)
        std  = np.nanstd(patch)
        out  = (patch - mean) / (std + 1e-8)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    else:
        raise ValueError(
            f"不支持的归一化方式: '{method}'。"
            f"可选: 'db_clip'（推荐）, 'minmax', 'percentile', 'zscore'"
        )

    denom = hi - lo
    if denom < 1e-10:
        return np.zeros_like(patch, dtype=np.float32)

    normalized = (patch - lo) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# 自定义异常
# ──────────────────────────────────────────────────────────────────

class DriveDisconnectedError(RuntimeError):
    """Google Drive FUSE 挂载断开时抛出，用于区分普通文件缺失错误。"""


# ──────────────────────────────────────────────────────────────────
# 主类：IcebergPreparer
# ──────────────────────────────────────────────────────────────────

class IcebergPreparer:
    """
    冰山数据集预处理流水线。
    负责将原始 SAR 影像 + .gpkg 矢量轮廓转换为可直接训练的 Patch 数据集。
    """

    def __init__(
        self,
        cfg,
        dry_run: bool = False,
        resume_enabled: Optional[bool] = None,
    ):
        """
        Args:
            cfg:     OmegaConf 配置对象（由 get_config() 返回）
            dry_run: 若为 True，则只打印统计信息，不写磁盘
        """
        self.cfg = cfg
        self.dry_run = dry_run
        self.dp = cfg.data_prep   # 预处理子配置的快捷引用
        self.paths = cfg.paths

        # 断点续跑：优先使用 CLI 显式参数，其次读取配置文件。
        cfg_resume = bool(self.dp.get("resume_enabled", False))
        self.resume_enabled = cfg_resume if resume_enabled is None else bool(resume_enabled)

        # 已完成场景记录文件（每行一个场景名）
        self._done_scenes_file = Path(self.paths.split_dir) / "completed_scenes.txt"
        self._done_scenes: set = set()

        self.patch_img_dir = Path(self.paths.patch_image_dir)
        self.patch_msk_dir = Path(self.paths.patch_mask_dir)
        self._existing_patch_index: Dict[str, List[Dict]] = {}

        # 目标坐标系
        self.target_crs = CRS.from_epsg(
            int(self.dp.target_crs.split(":")[1])
        )

        # 离线 RTV 平滑配置
        tv_cfg = self.dp.get("tv_smooth", {})
        self.tv_enabled = bool(
            tv_cfg.get("enabled", False) if isinstance(tv_cfg, dict) else tv_cfg.enabled
        )
        if self.tv_enabled:
            self.tv_weight    = float(tv_cfg.get("weight",    0.01))
            self.tv_sigma     = float(tv_cfg.get("sigma",     3.0))
            self.tv_sharpness = float(tv_cfg.get("sharpness", 0.005))
            self.tv_max_iter  = int(tv_cfg.get("max_iter",    4))
            logger.info(
                f"离线 RTV 平滑已启用: lambda={self.tv_weight}, "
                f"sigma={self.tv_sigma}, sharpness={self.tv_sharpness}, "
                f"iter={self.tv_max_iter}"
            )

        # 计数器（用于最终统计报告）
        self.stats = {
            "total_scenes": 0,
            "skipped_scenes_resume": 0,
            "total_patches": 0,
            "kept_patches": 0,
            "dropped_nan": 0,
            "dropped_no_iceberg": 0,
            "bg_patches_sampled": 0,
        }

    # ────────────────────────────────────────
    # 公共入口
    # ────────────────────────────────────────

    def run(self):
        """预处理流水线主入口。"""
        logger.info("=" * 60)

        if self.resume_enabled and not self.dry_run:
            self._done_scenes = self._load_done_scenes()
            logger.info(
                f"场景级断点续跑已启用，已完成场景: {len(self._done_scenes)} 个"
                f"（记录文件: {self._done_scenes_file}）"
            )
            if self._done_scenes:
                self._existing_patch_index = self._index_existing_scene_patches()

        existing_records_by_scene = self._load_existing_records_by_scene()
        logger.info("冰山数据集预处理开始")
        logger.info("=" * 60)

        # Step 1：扫描 SAR 文件列表
        sar_files = self._find_sar_files()
        if not sar_files:
            logger.error("在配置的所有 raw_sar_dirs 目录下未找到任何 .tif 文件，请检查路径。")
            return

        logger.info(f"共找到 {len(sar_files)} 个 SAR 场景")

        # Step 2：逐场景处理
        all_patch_records = []  # 收集所有 patch 的元信息
        resumed_scene_names: set = set()  # 记录 resume 跳过的场景，供划分时保留原分区

        for sar_path in sar_files:
            scene_name = sar_path.stem

            # 断点续跑：场景名在记录文件中则直接跳过，不做任何其他检测
            if self.resume_enabled and scene_name in self._done_scenes:
                self.stats["skipped_scenes_resume"] += 1
                self.stats["total_scenes"] += 1
                resumed_scene_names.add(scene_name)

                cached = existing_records_by_scene.get(scene_name)
                if cached:
                    self.stats["kept_patches"] += len(cached)
                    all_patch_records.extend(cached)
                    logger.info(
                        f"跳过已完成场景: {scene_name}（复用历史元数据 {len(cached)} 条）"
                    )
                else:
                    rebuilt = self._build_records_from_existing_patches(scene_name)
                    if rebuilt:
                        self.stats["kept_patches"] += len(rebuilt)
                        all_patch_records.extend(rebuilt)
                        logger.info(
                            f"跳过已完成场景: {scene_name}（重建元数据 {len(rebuilt)} 条）"
                        )
                    else:
                        logger.info(f"跳过已完成场景: {scene_name}（无有效切片）")
                continue

            try:
                records = self._process_scene(sar_path)
            except DriveDisconnectedError as e:
                logger.error(f"\n[!] Drive 挂载断开，预处理中断于场景: {scene_name}")
                logger.error(f"    {e}")
                logger.error(
                    "    处理方法：\n"
                    "    1. 在 Colab 重新运行 drive.mount('/content/drive', force_remount=True)\n"
                    "    2. 重新运行本脚本（已完成场景会自动跳过）"
                )
                break
            # 如果 _process_scene 返回 None，说明是因无法匹配文件等硬性错误跳过的，此时不标记
            if records is not None:
                all_patch_records.extend(records)
                # 无论 records 是否为空（即无论是否生成切片），都进行标记
                self._mark_scene_done(scene_name)

        logger.info(f"\n处理完成：共生成 {len(all_patch_records)} 个有效切片")

        # Step 3：生成数据集划分
        if not self.dry_run and all_patch_records:
            self._create_splits(all_patch_records, resumed_scene_names)

        # Step 4：打印统计报告
        self._print_stats()

    def _load_existing_records_by_scene(self) -> Dict[str, List[Dict]]:
        """
        读取历史 all_patches.csv，按场景聚合元数据。
        仅在启用 resume 且非 dry_run 时使用。
        """
        if not self.resume_enabled or self.dry_run:
            return {}

        all_csv = Path(self.paths.split_dir) / "all_patches.csv"
        if not all_csv.exists():
            return {}

        try:
            df = pd.read_csv(all_csv)
        except Exception as exc:
            logger.warning(f"读取历史元数据失败，将忽略 resume 元数据复用: {exc}")
            return {}

        if "scene" not in df.columns:
            logger.warning("all_patches.csv 缺少 scene 列，无法按场景复用元数据。")
            return {}

        grouped: Dict[str, List[Dict]] = {}
        for scene, sub in df.groupby("scene", sort=False):
            grouped[str(scene)] = sub.to_dict(orient="records")
        return grouped

    def _load_done_scenes(self) -> set:
        """从 completed_scenes.txt 加载已完成的场景名集合。"""
        if not self._done_scenes_file.exists():
            return set()
        try:
            text = self._done_scenes_file.read_text(encoding="utf-8")
            return {line.strip() for line in text.splitlines() if line.strip()}
        except Exception as exc:
            logger.warning(f"读取断点续跑记录失败，将从头处理所有场景: {exc}")
            return set()

    def _index_existing_scene_patches(self) -> Dict[str, List[Dict]]:
        """
        扫描已存在的 patch 文件，按 scene 建立索引。
        在 resume 模式下用于“只要已有 patch 就跳过整个场景”。
        """
        if not self.patch_img_dir.exists():
            return {}

        mask_names = set()
        if self.patch_msk_dir.exists():
            mask_names = {p.name for p in self.patch_msk_dir.glob("*.npy")}

        pattern = re.compile(r"^(?P<scene>.+)_r(?P<row>\d+)_c(?P<col>\d+)\.npy$")
        scene_to_patches: Dict[str, List[Dict]] = {}

        for img_path in self.patch_img_dir.glob("*.npy"):
            m = pattern.match(img_path.name)
            if not m:
                continue
            if img_path.name not in mask_names:
                continue

            scene_name = m.group("scene")
            scene_to_patches.setdefault(scene_name, []).append(
                {
                    "row_offset": int(m.group("row")),
                    "col_offset": int(m.group("col")),
                    "image_path": str(img_path),
                    "mask_path": str(self.patch_msk_dir / img_path.name),
                }
            )

        return scene_to_patches

    def _build_records_from_existing_patches(self, scene_name: str) -> List[Dict]:
        """
        当 all_patches.csv 缺失场景记录时，从已有 patch 文件名重建最小元数据。
        """
        entries = self._existing_patch_index.get(scene_name, [])
        if not entries:
            return []

        sorted_entries = sorted(entries, key=lambda x: (x["row_offset"], x["col_offset"]))
        records: List[Dict] = []
        for patch_id, e in enumerate(sorted_entries):
            records.append(
                {
                    "scene": scene_name,
                    "patch_id": patch_id,
                    "row_offset": int(e["row_offset"]),
                    "col_offset": int(e["col_offset"]),
                    "image_path": e["image_path"],
                    "mask_path": e["mask_path"],
                    # 由文件名重建时无法无损恢复统计字段，使用占位值。
                    "iceberg_pixels": -1,
                    "nan_ratio": np.nan,
                }
            )
        return records

    def _mark_scene_done(self, scene_name: str):
        """将场景名追加到 completed_scenes.txt，标记为已完成。"""
        if not self.resume_enabled or self.dry_run:
            return
        self._done_scenes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._done_scenes_file, "a", encoding="utf-8") as f:
            f.write(scene_name + "\n")
        self._done_scenes.add(scene_name)

    # ────────────────────────────────────────
    # Step 1：扫描 SAR 文件
    # ────────────────────────────────────────

    def _find_sar_files(self) -> List[Path]:
        """扫描所有 raw_sar_dirs 目录，返回去重后的 .tif 文件列表。
        兼容单字符串（raw_sar_dir）和列表（raw_sar_dirs）两种配置写法。
        """
        raw = self.paths.get("raw_sar_dirs", None) or self.paths.get("raw_sar_dir", None)
        if raw is None:
            raise ValueError("config.yaml 中未找到 raw_sar_dirs 或 raw_sar_dir 配置项")

        # 统一转为列表
        dirs = [raw] if isinstance(raw, str) else list(raw)

        seen: set = set()
        files: List[Path] = []
        for d in dirs:
            sar_dir = Path(d)
            if not sar_dir.exists():
                logger.warning(f"SAR 目录不存在，跳过: {sar_dir}")
                continue
            for f in sorted(sar_dir.iterdir()):
                if f.suffix.lower() == ".tif" and f.name not in seen:
                    seen.add(f.name)
                    files.append(f)
        return files

    def _find_matching_vector(self, year_month: str) -> Optional[Path]:
        """
        根据年月字符串（如 '201810'）在 raw_vector_dir 中找对应 .gpkg 文件。
        命名约定：{YYYYMM}_distribution.gpkg
        """
        vec_dir = Path(self.paths.raw_vector_dir)
        # 精确匹配
        exact = vec_dir / f"{year_month}_distribution.gpkg"
        try:
            if exact.exists():
                return exact
            # 模糊匹配：文件名中包含年月
            for f in vec_dir.glob("*.gpkg"):
                if year_month in f.stem:
                    return f
        except OSError as e:
            import errno as _errno
            if e.errno == _errno.ENOTCONN:  # 107: Transport endpoint is not connected
                raise DriveDisconnectedError(
                    "Google Drive 挂载断开 (errno=107)。"
                    "请在 Colab 重新运行挂载单元格后，再次执行脚本（断点续跑会自动跳过已完成场景）。"
                ) from e
            raise
        return None

    # ────────────────────────────────────────
    # Step 2：单场景处理
    # ────────────────────────────────────────

    def _process_scene(self, sar_path: Path) -> Optional[List[Dict]]:
        """
        处理单个 SAR 场景：重投影 → 烧录掩膜 → 切图 → 保存。

        Returns:
            - 成功处理并生成切片: 返回 patch 元信息列表 (List[Dict])
            - 成功处理但无冰山/切片: 返回空列表 []
            - 因文件匹配等硬性错误跳过: 返回 None
        """
        logger.info(f"\n{'─'*50}")
        logger.info(f"处理场景: {sar_path.name}")

        # ---- 匹配矢量文件 ----
        year_month = extract_year_month(sar_path)
        if year_month is None:
            logger.warning(f"  无法从文件名中解析年月，跳过: {sar_path.name}")
            return None

        vec_path = self._find_matching_vector(year_month)
        if vec_path is None:
            logger.warning(f"  未找到年月 {year_month} 对应的矢量文件，跳过。")
            return None

        logger.info(f"  匹配矢量: {vec_path.name}")
        self.stats["total_scenes"] += 1

        # ---- 重投影 SAR 到目标 CRS ----
        try:
            sar_reprojected, sar_transform, sar_crs = self._reproject_sar(sar_path)
            logger.info(
                f"  重投影后尺寸: {sar_reprojected.shape[1]} x {sar_reprojected.shape[0]}，"
                f"CRS: {sar_crs.to_epsg()}"
            )
        except Exception as e:
            logger.error(f"  重投影失败，跳过场景: {sar_path.name}\n  错误: {e}")
            return None

        # ---- 加载矢量并烧录为掩膜 ----
        mask = self._build_mask(
            vec_path=vec_path,
            raster_shape=(sar_reprojected.shape[0], sar_reprojected.shape[1]),
            raster_transform=sar_transform,
            raster_crs=sar_crs,
        )
        iceberg_pixels = int((mask > 0).sum())   # 前景像素数（ID > 0）
        logger.info(f"  掩膜中冰山像素数: {iceberg_pixels:,}")

        if iceberg_pixels == 0:
            # _build_mask 内已打印详细诊断，此处不再重复
            # 即使没有冰山，也返回空列表表示“已处理但无结果”
            return []

        # ---- 切图并保存 ----
        scene_name = sar_path.stem
        records = self._extract_and_save_patches(
            image=sar_reprojected,
            mask=mask,
            scene_name=scene_name,
            transform=sar_transform,
            crs=sar_crs,
        )
        logger.info(f"  保留切片: {len(records)} / {self.stats['total_patches']} 总计")
        return records

    # ────────────────────────────────────────
    # 重投影 SAR
    # ────────────────────────────────────────

    def _reproject_sar(
        self, sar_path: Path
    ) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
        """
        将 SAR GeoTIFF 重投影到目标坐标系（EPSG:3031）。

        Returns:
            (image_array, transform, crs)
            image_array: shape (H, W)，float32，NaN 表示无数据区域
        """
        band_idx = self.dp.sar_bands[0]  # 通常只用第一个波段

        with rasterio.open(sar_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_nodata = src.nodata
            src_dtype = src.dtypes[band_idx - 1]

            # 计算重投影后的仿射变换和输出尺寸
            target_res = self.dp.get("target_resolution", None)
            if target_res is not None:
                # 指定目标分辨率（单位：米）
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, self.target_crs,
                    src.width, src.height,
                    *src.bounds,
                    resolution=target_res,
                )
            else:
                # 保持原始像素数量
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, self.target_crs,
                    src.width, src.height,
                    *src.bounds,
                )

            # 申请输出数组（float32，NaN 填充无数据）
            dst_array = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

            # 执行重投影（双线性插值）
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_array,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=self.target_crs,
                resampling=Resampling.bilinear,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
            )

        return dst_array, dst_transform, self.target_crs

    # ────────────────────────────────────────
    # 矢量烧录（rasterize）
    # ────────────────────────────────────────

    def _build_mask(
        self,
        vec_path: Path,
        raster_shape: Tuple[int, int],
        raster_transform: rasterio.Affine,
        raster_crs: CRS,
    ) -> np.ndarray:
        """
        加载 .gpkg 矢量，重投影到栅格坐标系，
        按空间范围过滤后烧录为二值掩膜。

        Returns:
            shape (H, W)，dtype uint8，0=背景，1=冰山
        """
        H, W = raster_shape

        # ---- 计算栅格在目标 CRS 中的空间范围 ----
        from rasterio.transform import array_bounds
        left, bottom, right, top = array_bounds(H, W, raster_transform)
        raster_bbox = box(left, bottom, right, top)

        logger.info(
            f"  栅格范围 (EPSG:{raster_crs.to_epsg()}): "
            f"X=[{left:.0f}, {right:.0f}], Y=[{bottom:.0f}, {top:.0f}]"
        )

        # ---- 读取矢量 ----
        logger.info(f"  加载矢量文件: {vec_path.name} ...")
        gdf = gpd.read_file(vec_path)

        # ---- 重投影矢量到栅格 CRS ----
        vec_epsg = gdf.crs.to_epsg()
        raster_epsg = raster_crs.to_epsg()
        if vec_epsg != raster_epsg:
            logger.info(f"  矢量 CRS (EPSG:{vec_epsg}) → 栅格 CRS (EPSG:{raster_epsg})")
            gdf = gdf.to_crs(raster_crs)

        # ---- 空间过滤：只保留与栅格范围（+可选缓冲）相交的要素 ----
        buffer_m = float(self.dp.get("spatial_buffer_m", 0))
        filter_geom = raster_bbox.buffer(buffer_m) if buffer_m > 0 else raster_bbox

        original_count = len(gdf)
        gdf = gdf[gdf.geometry.intersects(filter_geom)].copy()
        logger.info(
            f"  空间过滤 (buffer={buffer_m:.0f}m): "
            f"{original_count} → {len(gdf)} 个要素"
        )

        if len(gdf) == 0:
            # ── 诊断信息：找出最近的矢量要素，帮助判断数据是否错位 ──
            gdf_all = gpd.read_file(vec_path).to_crs(raster_crs)
            distances = gdf_all.geometry.distance(raster_bbox)
            min_dist = distances.min()
            nearest = gdf_all.loc[distances.idxmin()]
            logger.warning(
                f"  SAR 范围内无冰山矢量。\n"
                f"  最近要素距离: {min_dist/1000:.1f} km  "
                f"(lon={nearest.get('lon', '?'):.3f}, lat={nearest.get('lat', '?'):.3f})\n"
                f"  可能原因: ① 该 SAR 瓦片覆盖的海域冰山稀少；"
                f"② 年份/月份不匹配；③ CRS 转换异常。\n"
                f"  提示: 若为测试目的，可在 config.yaml 中临时设置 "
                f"spatial_buffer_m: {int(min_dist)+1000} 以包含最近的冰山。"
            )
            return np.zeros((H, W), dtype=np.uint8)

        # ---- 展平复杂几何，并保留每个要素的原始实例 ID ----
        # feature_id 从 1 开始，同一个 .gpkg 要素的所有部件共享同一 ID
        # （MultiPolygon 拆分后各部件仍属同一冰山实例）
        shape_list = []   # List of (geometry_mapping, instance_id)
        skipped = 0
        for feature_id, geom in enumerate(gdf.geometry, start=1):
            polys = flatten_geometry(geom)
            if not polys:
                skipped += 1
                continue
            for poly in polys:
                # ---- 裁剪到栅格范围（防止越界）----
                clipped = poly.intersection(raster_bbox)
                for c in flatten_geometry(clipped):
                    shape_list.append((mapping(c), feature_id))

        if skipped:
            logger.info(f"  跳过 {skipped} 个无效/空几何体")

        if not shape_list:
            logger.warning("  展平后无有效多边形，返回空掩膜。")
            return np.zeros((H, W), dtype=np.uint16)

        # ---- 烧录：每个多边形赋予唯一整数 ID（非全部烧成 1）----
        # dtype=uint16：支持单场景最多 65535 个冰山实例
        # all_touched=False：只烧录中心点在多边形内的像素（保证边界精度）
        mask = rasterize(
            shapes=shape_list,
            out_shape=(H, W),
            transform=raster_transform,
            fill=0,           # 背景 = 0
            dtype=np.uint16,
            all_touched=False,
        )

        n_instances = len(np.unique(mask)) - 1  # 减去背景 0
        logger.info(f"  烧录完成: {n_instances} 个冰山实例 ID 写入掩膜")
        return mask

    # ────────────────────────────────────────
    # 切图与保存
    # ────────────────────────────────────────

    def _extract_and_save_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        scene_name: str,
        transform: rasterio.Affine,
        crs: CRS,
    ) -> List[Dict]:
        """
        对整幅图像和掩膜进行滑窗切片，过滤无效切片后保存到磁盘。

        Returns:
            保留的切片元信息列表（每个字典包含 scene, patch_id, image_path, mask_path）
        """
        H, W = image.shape
        patch_size = int(self.dp.patch_size)
        overlap = int(self.dp.overlap)
        stride = patch_size - overlap

        img_out_dir = Path(self.paths.patch_image_dir)
        msk_out_dir = Path(self.paths.patch_mask_dir)
        img_out_dir.mkdir(parents=True, exist_ok=True)
        msk_out_dir.mkdir(parents=True, exist_ok=True)

        bg_ratio = float(self.dp.get("background_patch_ratio", 0.0))

        records = []
        patch_id = 0
        bg_candidate_positions: List[Tuple[int, int]] = []  # (row, col) of valid bg patches

        # 滑窗遍历：行优先
        row_starts = list(range(0, H - patch_size + 1, stride))
        col_starts = list(range(0, W - patch_size + 1, stride))

        # 确保覆盖末尾区域
        if row_starts and (row_starts[-1] + patch_size < H):
            row_starts.append(H - patch_size)
        if col_starts and (col_starts[-1] + patch_size < W):
            col_starts.append(W - patch_size)

        total = len(row_starts) * len(col_starts)
        self.stats["total_patches"] += total

        pbar = tqdm(
            total=total,
            desc=f"  切片 [{scene_name}]",
            unit="patch",
            ncols=80,
        )

        for row in row_starts:
            for col in col_starts:
                pbar.update(1)

                img_patch = image[row: row + patch_size, col: col + patch_size]
                msk_patch = mask[row: row + patch_size, col: col + patch_size]

                # ---- 有效性检查 ----
                drop_reason = self._get_drop_reason(img_patch, msk_patch)
                if drop_reason == "nan":
                    self.stats["dropped_nan"] += 1
                    continue
                if drop_reason == "no_iceberg":
                    # 暂存位置，待前景统计完成后按比例随机采样
                    if bg_ratio > 0:
                        bg_candidate_positions.append((row, col))
                    continue

                self.stats["kept_patches"] += 1

                # ---- 离线 RTV 平滑（在归一化之前，在 dB 域操作）----
                # 对应论文 MATLAB: db2mag → tsmooth → (存储) → 归一化
                if self.tv_enabled:
                    img_patch = rtv_smooth_db(
                        img_patch,
                        weight=self.tv_weight,
                        sigma=self.tv_sigma,
                        sharpness=self.tv_sharpness,
                        max_iter=self.tv_max_iter,
                    )

                # ---- 归一化 ----
                norm_patch = normalize_sar(
                    img_patch,
                    method=self.dp.normalize.method,
                    db_min=float(self.dp.normalize.db_clip_min),
                    db_max=float(self.dp.normalize.db_clip_max),
                    p_low=float(self.dp.normalize.percentile_low),
                    p_high=float(self.dp.normalize.percentile_high),
                )

                # ---- 保存（.npy 格式，IO 最快，无地理信息依赖）----
                fname = f"{scene_name}_r{row:05d}_c{col:05d}"
                img_save_path = img_out_dir / f"{fname}.npy"
                msk_save_path = msk_out_dir / f"{fname}.npy"

                if not self.dry_run:
                    np.save(img_save_path, norm_patch)       # (H, W) float32
                    np.save(msk_save_path, msk_patch)        # (H, W) uint16，每像素为冰山实例 ID

                records.append({
                    "scene":      scene_name,
                    "patch_id":   patch_id,
                    "row_offset": row,
                    "col_offset": col,
                    "image_path": str(img_save_path),
                    "mask_path":  str(msk_save_path),
                    "iceberg_pixels": int((msk_patch > 0).sum()),
                    "nan_ratio":  float(np.isnan(img_patch).mean()),
                })
                patch_id += 1

        pbar.close()

        # ---- 背景切片采样 ----
        n_bg_candidates = len(bg_candidate_positions)
        n_fg = len(records)
        n_bg_target = min(round(n_fg * bg_ratio), n_bg_candidates)

        if n_bg_target > 0:
            rng = np.random.default_rng(int(self.dp.random_seed))
            chosen_idx = rng.choice(n_bg_candidates, size=n_bg_target, replace=False)

            for idx in sorted(chosen_idx.tolist()):
                row, col = bg_candidate_positions[idx]
                img_patch = image[row: row + patch_size, col: col + patch_size]
                msk_patch = mask[row: row + patch_size, col: col + patch_size]  # 全零掩膜

                norm_patch = normalize_sar(
                    img_patch,
                    method=self.dp.normalize.method,
                    db_min=float(self.dp.normalize.db_clip_min),
                    db_max=float(self.dp.normalize.db_clip_max),
                    p_low=float(self.dp.normalize.percentile_low),
                    p_high=float(self.dp.normalize.percentile_high),
                )

                fname = f"{scene_name}_r{row:05d}_c{col:05d}"
                img_save_path = img_out_dir / f"{fname}.npy"
                msk_save_path = msk_out_dir / f"{fname}.npy"

                if not self.dry_run:
                    np.save(img_save_path, norm_patch)
                    np.save(msk_save_path, msk_patch)

                records.append({
                    "scene":      scene_name,
                    "patch_id":   patch_id,
                    "row_offset": row,
                    "col_offset": col,
                    "image_path": str(img_save_path),
                    "mask_path":  str(msk_save_path),
                    "iceberg_pixels": 0,
                    "nan_ratio":  float(np.isnan(img_patch).mean()),
                })
                patch_id += 1

        n_bg_sampled = n_bg_target
        self.stats["dropped_no_iceberg"] += n_bg_candidates - n_bg_sampled
        self.stats["bg_patches_sampled"] += n_bg_sampled
        self.stats["kept_patches"] += n_bg_sampled

        return records

    def _get_drop_reason(
        self, img_patch: np.ndarray, msk_patch: np.ndarray
    ) -> Optional[str]:
        """
        检查切片是否应被丢弃。
        返回丢弃原因字符串，或 None（保留）。
        """
        # 规则 1：NaN 比例超过阈值
        nan_ratio = float(np.isnan(img_patch).mean())
        if nan_ratio > float(self.dp.max_nan_ratio):
            return "nan"

        # 规则 2：冰山像素数量不足（ID > 0 的前景像素计数）
        if int((msk_patch > 0).sum()) < int(self.dp.min_iceberg_pixels):
            return "no_iceberg"

        return None

    # ────────────────────────────────────────
    # 数据集划分
    # ────────────────────────────────────────

    def _create_splits(self, records: List[Dict], resumed_scene_names=None):
        """
        按配置比例将所有 patch 随机划分为 train / val / test，
        保存为 CSV 文件至 split_dir。

        resumed_scene_names: resume 跳过的场景集合。
            这些场景已在历史 all_patches.csv 中有分区记录，直接复用，
            不参与重新 shuffle，防止增量添加数据时旧场景分区发生变动。
        """
        rng = np.random.default_rng(int(self.dp.random_seed))
        df = pd.DataFrame(records)

        train_ratio = float(self.dp.split_ratio.train)
        val_ratio   = float(self.dp.split_ratio.val)

        resumed_scene_names = set(resumed_scene_names or [])

        # ---- 读取历史分区（仅对 resume 场景）----
        scene_to_old_split: Dict[str, str] = {}
        if resumed_scene_names:
            all_csv = Path(self.paths.split_dir) / "all_patches.csv"
            if all_csv.exists():
                try:
                    old_df = pd.read_csv(all_csv, usecols=["scene", "split"])
                    if "split" in old_df.columns:
                        tmp = (
                            old_df[old_df["scene"].isin(resumed_scene_names)]
                            .drop_duplicates("scene")
                        )
                        scene_to_old_split = dict(
                            zip(tmp["scene"], tmp["split"])
                        )
                except Exception as exc:
                    logger.warning(f"读取历史分区记录失败，将重新划分所有场景: {exc}")

        # ---- 只对新场景做 shuffle 和分区 ----
        all_scenes = df["scene"].unique()
        new_scenes = np.array([s for s in all_scenes if s not in scene_to_old_split])
        rng.shuffle(new_scenes)

        n = len(new_scenes)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        split_map: Dict[str, str] = dict(scene_to_old_split)  # 先填入旧场景分区
        for s in new_scenes[:n_train]:
            split_map[s] = "train"
        for s in new_scenes[n_train: n_train + n_val]:
            split_map[s] = "val"
        for s in new_scenes[n_train + n_val:]:
            split_map[s] = "test"

        df["split"] = df["scene"].map(split_map)

        if scene_to_old_split:
            logger.info(
                f"  分区复用: {len(scene_to_old_split)} 个旧场景保持原分区；"
                f"{n} 个新场景重新划分"
            )

        # 若新场景仅 1 个且无历史分区，退化为按切片随机划分
        if n == 1 and not scene_to_old_split:
            logger.warning(
                "只有 1 个 SAR 场景，改为按切片随机划分（训练与验证集可能有空间重叠）。"
            )
            indices = np.arange(len(df))
            rng.shuffle(indices)
            n_p = len(df)
            n_train_p = int(n_p * train_ratio)
            n_val_p   = int(n_p * val_ratio)

            splits = ["train"] * n_train_p + ["val"] * n_val_p + \
                     ["test"] * (n_p - n_train_p - n_val_p)
            df["split"] = [splits[i] for i in np.argsort(indices)]

        split_dir = Path(self.paths.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整元数据
        all_csv = split_dir / "all_patches.csv"
        df.to_csv(all_csv, index=False)
        logger.info(f"\n数据集划分已保存: {all_csv}")

        # 分别保存各子集
        for split_name in ["train", "val", "test"]:
            sub = df[df["split"] == split_name]
            out = split_dir / f"{split_name}.csv"
            sub.to_csv(out, index=False)
            logger.info(f"  {split_name:5s}: {len(sub):5d} 个切片  → {out.name}")

    # ────────────────────────────────────────
    # 统计报告
    # ────────────────────────────────────────

    def _print_stats(self):
        s = self.stats
        logger.info("\n" + "=" * 60)
        logger.info("预处理统计报告")
        logger.info("=" * 60)
        logger.info(f"  处理场景数:        {s['total_scenes']}")
        logger.info(f"  Resume跳过场景:    {s['skipped_scenes_resume']}")
        logger.info(f"  切片总数:          {s['total_patches']}")
        logger.info(f"  丢弃（NaN过多）:    {s['dropped_nan']}")
        logger.info(f"  丢弃（无冰山）:     {s['dropped_no_iceberg']}")
        logger.info(f"  背景切片采样:      {s['bg_patches_sampled']}")
        logger.info(f"  最终保留:          {s['kept_patches']}"
                    f"（前景 {s['kept_patches'] - s['bg_patches_sampled']} + "
                    f"背景 {s['bg_patches_sampled']}）")
        if s["total_patches"] > 0:
            keep_rate = s["kept_patches"] / s["total_patches"] * 100
            logger.info(f"  保留率:            {keep_rate:.1f}%")
        logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="南极冰山数据集预处理脚本"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅统计，不写磁盘（用于快速验证参数）",
    )

    # CLI 显式覆盖配置中的 resume_enabled
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume_enabled",
        action="store_true",
        help="启用场景级断点续跑（跳过已完成场景）",
    )
    resume_group.add_argument(
        "--no_resume",
        dest="resume_enabled",
        action="store_false",
        help="禁用场景级断点续跑",
    )
    parser.set_defaults(resume_enabled=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 切换到项目根目录，使相对路径正常工作
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    cfg = get_config(args.config)
    preparer = IcebergPreparer(
        cfg,
        dry_run=args.dry_run,
        resume_enabled=args.resume_enabled,
    )
    preparer.run()
