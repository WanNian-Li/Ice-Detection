"""
data_prep/export_scene_meta.py
==============================
从原始 SAR .tif 文件提取每个场景重投影后的地理变换元数据，
保存为 JSON 供后续地理边界过滤使用。

输出 JSON 结构（每个 key 为场景名 stem，即不含扩展名的文件名）：
    {
        "grid_1_hh_2018_10": {
            "origin_x":  -2800000.0,   # 重投影后左上角 X (EPSG:3031, 单位: 米)
            "origin_y":   2100000.0,   # 重投影后左上角 Y (EPSG:3031, 单位: 米)
            "res_x":           40.0,   # 像素宽度 (米, 正值)
            "res_y":          -40.0,   # 像素高度 (米, 负值，Y 轴朝下)
            "width":          12800,   # 重投影后总列数
            "height":          9600,   # 重投影后总行数
            "crs_epsg":        3031    # 坐标系 EPSG 代码
        },
        ...
    }

有了此 JSON，给定 HDF5 中记录的 (scene, row_offset, col_offset) 可直接计算
任意 patch 在 EPSG:3031 中的地理范围：
    patch_left   = origin_x + col_offset * res_x
    patch_top    = origin_y + row_offset * res_y
    patch_right  = patch_left  + patch_size * res_x
    patch_bottom = patch_top   + patch_size * res_y

使用方式（在 Colab 上运行）：
    # 挂载 Drive 后运行
    python data_prep/export_scene_meta.py --config configs/config.yaml

    # 指定输出路径（默认写到 data/scene_meta.json，建议 git commit）
    python data_prep/export_scene_meta.py --output /content/drive/MyDrive/scene_meta.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 工具函数（与 prepare_dataset.py 保持一致）
# ──────────────────────────────────────────────────────────────────

def find_sar_files(cfg) -> List[Path]:
    """扫描所有 raw_sar_dirs 目录，返回去重后的 .tif 文件列表。"""
    raw = cfg.paths.get("raw_sar_dirs", None) or cfg.paths.get("raw_sar_dir", None)
    if raw is None:
        raise ValueError("config.yaml 中未找到 raw_sar_dirs 或 raw_sar_dir 配置项")

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


def get_reprojected_transform(
    tif_path: Path,
    target_crs: CRS,
    target_resolution: Optional[float],
) -> Optional[Dict]:
    """
    只读取 .tif 元数据（不加载像素），计算重投影后的仿射变换。

    与 prepare_dataset._reproject_sar() 中的 calculate_default_transform
    调用完全一致，保证与实际切图时的坐标系对齐。

    Returns:
        dict 或 None（读取失败时）
    """
    try:
        with rasterio.open(tif_path) as src:
            src_crs = src.crs

            if target_resolution is not None:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, target_crs,
                    src.width, src.height,
                    *src.bounds,
                    resolution=target_resolution,
                )
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, target_crs,
                    src.width, src.height,
                    *src.bounds,
                )

        return {
            "origin_x":  float(dst_transform.c),   # 左上角 X
            "origin_y":  float(dst_transform.f),   # 左上角 Y
            "res_x":     float(dst_transform.a),   # 像素宽（正）
            "res_y":     float(dst_transform.e),   # 像素高（负，Y 轴朝下）
            "width":     int(dst_width),
            "height":    int(dst_height),
            "crs_epsg":  int(target_crs.to_epsg()),
        }

    except Exception as e:
        logger.warning(f"读取失败，跳过 {tif_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从原始 SAR .tif 提取重投影元数据，保存为 JSON"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output", default=None,
        help="输出 JSON 路径（默认: data/scene_meta.json）"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    cfg = get_config(args.config)

    target_crs = CRS.from_epsg(
        int(cfg.data_prep.target_crs.split(":")[1])
    )
    target_res = cfg.data_prep.get("target_resolution", None)
    if target_res is not None:
        target_res = float(target_res)

    out_path = Path(args.output) if args.output else Path("data/scene_meta.json")

    logger.info("=" * 60)
    logger.info("场景地理元数据提取")
    logger.info(f"  目标 CRS   : EPSG:{target_crs.to_epsg()}")
    logger.info(f"  目标分辨率 : {target_res} m")
    logger.info(f"  输出路径   : {out_path}")
    logger.info("=" * 60)

    tif_files = find_sar_files(cfg)
    if not tif_files:
        logger.error("未找到任何 .tif 文件，请检查 config.yaml 中的 raw_sar_dirs。")
        return

    logger.info(f"共找到 {len(tif_files)} 个 SAR 场景，开始提取元数据...")

    scene_meta: Dict[str, Dict] = {}
    n_ok = 0
    n_fail = 0

    for tif_path in tqdm(tif_files, desc="提取", unit="scene", ncols=80):
        scene_name = tif_path.stem
        meta = get_reprojected_transform(tif_path, target_crs, target_res)
        if meta is not None:
            scene_meta[scene_name] = meta
            n_ok += 1
        else:
            n_fail += 1

    # 输出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"完成: 成功 {n_ok} 个，失败 {n_fail} 个")
    logger.info(f"已保存: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    logger.info("=" * 60)

    # 打印第一条记录作为样例
    if scene_meta:
        first_name, first_meta = next(iter(scene_meta.items()))
        logger.info(f"\n样例记录 [{first_name}]:")
        for k, v in first_meta.items():
            logger.info(f"  {k}: {v}")

    logger.info(
        "\n下一步：\n"
        "  1. 将 scene_meta.json 提交到 git 或上传到服务器\n"
        "  2. 下载 MEaSUREs Antarctic Boundaries shapefile\n"
        "     https://nsidc.org/data/nsidc-0709\n"
        "  3. 运行 evaluate.py --geo_boundary_shp <路径> --scene_meta data/scene_meta.json"
    )


if __name__ == "__main__":
    main()
