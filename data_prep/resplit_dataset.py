"""
data_prep/resplit_dataset.py
============================
对 data/processed/images 下已处理完毕的切片重新划分 train/val/test。

适用场景：分批次（按年份）运行 prepare_dataset.py 后，split_dir 中的 CSV
只包含最后一批的切片记录，需要对全量已处理切片重新生成一致的划分。
默认只会重划分与 configs/config.yaml 中 paths.raw_sar_dirs 对应年份一致的场景。

逻辑：
  1. 扫描 patch_image_dir 下所有 .npy 文件，从文件名解析 scene/row/col
  2. 验证对应的 mask 文件存在于 patch_mask_dir
  3. 按 SAR 场景分组，随机打乱后按比例划分（防止空间泄露）
  4. 写出 all_patches.csv + train.csv + val.csv + test.csv 到 split_dir

使用方式：
    python data_prep/resplit_dataset.py --config configs/config.yaml
    python data_prep/resplit_dataset.py --config configs/config.yaml --seed 42
    python data_prep/resplit_dataset.py --config configs/config.yaml --dry_run
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 与 prepare_dataset.py 保持一致的文件名模式
PATCH_PATTERN = re.compile(r"^(?P<scene>.+)_r(?P<row>\d+)_c(?P<col>\d+)\.npy$")
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _extract_years_from_raw_dirs(raw_sar_dirs) -> set[str]:
    """从 raw_sar_dirs 的目录名中提取 4 位年份白名单。"""
    years: set[str] = set()
    for raw_dir in raw_sar_dirs or []:
        name = Path(raw_dir).name
        match = YEAR_PATTERN.search(name)
        if match:
            years.add(match.group(0))
    return years


def _extract_year_from_scene(scene: str) -> str | None:
    """从 patch scene 名称中提取 4 位年份。"""
    match = YEAR_PATTERN.search(scene)
    return match.group(0) if match else None


def _list_npy_names(dir_str: str, label: str) -> list[str]:
    """
    列出目录下所有 .npy 文件名（不含路径）。

    Google Drive FUSE 挂载大目录时，os.scandir() / os.listdir() 会抛出
    OSError [Errno 5]。用 subprocess + find 绕过该限制；失败时重试一次后
    再回退到 os.listdir()。
    """
    for attempt in range(2):
        try:
            result = subprocess.run(
                ["find", dir_str, "-maxdepth", "1", "-name", "*.npy"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                names = [
                    os.path.basename(p)
                    for p in result.stdout.splitlines()
                    if p.strip().endswith(".npy")
                ]
                logger.info(f"{label} 目录共找到 {len(names)} 个 .npy 文件（find）")
                return names
            logger.warning(f"find 返回非零退出码 {result.returncode}，重试...")
        except subprocess.TimeoutExpired:
            logger.warning(f"find 超时（{label}），重试...")
        except OSError as e:
            logger.warning(f"find 调用失败（{label}）: {e}，重试...")
        if attempt == 0:
            time.sleep(3)

    # 最终回退：os.listdir()
    logger.warning(f"find 失败，回退到 os.listdir()（{label}）")
    try:
        names = [n for n in os.listdir(dir_str) if n.endswith(".npy")]
        logger.info(f"{label} 目录共找到 {len(names)} 个 .npy 文件（listdir）")
        return names
    except OSError as e:
        logger.error(f"os.listdir() 也失败（{label}）: {e}")
        return []


def scan_patches(img_dir: Path, msk_dir: Path, allowed_years: set[str] | None = None) -> list[dict]:
    """
    扫描 img_dir 下所有 .npy 文件，验证对应 mask 存在后构建 patch 记录列表。

    使用 subprocess+find 替代 os.scandir()，解决 Google Drive FUSE 挂载
    大目录时抛出 OSError [Errno 5] 的已知问题。mask 文件名预加载为集合，
    避免逐文件网络 exists() 查询。

    Returns:
        每条记录含: scene, patch_id(场景内序号), row_offset, col_offset,
                    image_path, mask_path
    """
    img_dir_str = str(img_dir)
    msk_dir_str = str(msk_dir)

    if not os.path.isdir(img_dir_str):
        raise FileNotFoundError(f"patch_image_dir 不存在: {img_dir_str}")
    if not os.path.isdir(msk_dir_str):
        raise FileNotFoundError(f"patch_mask_dir 不存在: {msk_dir_str}")

    logger.info("预加载 mask 文件列表 ...")
    mask_names: set[str] = set(_list_npy_names(msk_dir_str, "mask"))

    logger.info("扫描 image 目录 ...")
    img_names = _list_npy_names(img_dir_str, "image")

    if not img_names:
        logger.error(
            "image 目录内无 .npy 文件，请确认路径正确且文件已上传。\n"
            f"  当前路径: {img_dir_str}"
        )
        return []

    # 按场景聚合
    scene_patches: dict[str, list[dict]] = {}
    bad_name = 0
    missing_masks = 0
    filtered_years = 0

    for name in img_names:
        m = PATCH_PATTERN.match(name)
        if not m:
            bad_name += 1
            if bad_name <= 5:
                logger.warning(f"文件名不符合命名规范，跳过: {name}")
            continue

        if name not in mask_names:
            missing_masks += 1
            if missing_masks <= 5:
                logger.warning(f"缺少对应 mask，跳过: {name}")
            continue

        scene = m.group("scene")
        if allowed_years:
            scene_year = _extract_year_from_scene(scene)
            if scene_year not in allowed_years:
                filtered_years += 1
                if filtered_years <= 5:
                    logger.info(f"场景年份不在配置白名单内，跳过: {name} (year={scene_year})")
                continue

        row   = int(m.group("row"))
        col   = int(m.group("col"))

        scene_patches.setdefault(scene, []).append({
            "scene":      scene,
            "row_offset": row,
            "col_offset": col,
            "image_path": os.path.join(img_dir_str, name),
            "mask_path":  os.path.join(msk_dir_str, name),
        })

    if bad_name:
        logger.warning(f"共跳过 {bad_name} 个文件名不规范的图像 patch")
    if missing_masks:
        logger.warning(f"共跳过 {missing_masks} 个缺少 mask 的图像 patch")
    if filtered_years:
        logger.warning(f"共跳过 {filtered_years} 个不在 raw_sar_dirs 年份白名单内的图像 patch")

    # 场景内按 (row, col) 排序后编号
    records = []
    for scene in sorted(scene_patches):
        patches = sorted(scene_patches[scene], key=lambda x: (x["row_offset"], x["col_offset"]))
        for patch_id, p in enumerate(patches):
            p["patch_id"] = patch_id
            records.append(p)

    return records


def make_splits(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    按场景随机划分 train/val/test，将 split 列写入 DataFrame 并返回。

    当场景数为 1 时退化为按切片随机划分（并打印警告）。
    """
    df = pd.DataFrame(records)
    rng = np.random.default_rng(seed)

    scenes = df["scene"].unique()
    rng.shuffle(scenes)
    n = len(scenes)

    if n == 1:
        logger.warning(
            "只检测到 1 个 SAR 场景，改为按切片随机划分（训练与验证集可能有空间重叠）。"
        )
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = int(len(df) * train_ratio)
        n_val   = int(len(df) * val_ratio)
        splits  = (["train"] * n_train
                   + ["val"]   * n_val
                   + ["test"]  * (len(df) - n_train - n_val))
        df["split"] = [splits[i] for i in np.argsort(indices)]
        return df

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    split_map: dict[str, str] = {}
    for s in scenes[:n_train]:
        split_map[s] = "train"
    for s in scenes[n_train: n_train + n_val]:
        split_map[s] = "val"
    for s in scenes[n_train + n_val:]:
        split_map[s] = "test"

    df["split"] = df["scene"].map(split_map)
    return df


def save_splits(df: pd.DataFrame, split_dir: Path, dry_run: bool):
    """将 all_patches.csv 及各子集 CSV 写到 split_dir。"""
    if dry_run:
        logger.info("[dry_run] 不写磁盘，仅打印统计。")
        return

    split_dir.mkdir(parents=True, exist_ok=True)

    all_csv = split_dir / "all_patches.csv"
    df.to_csv(all_csv, index=False)
    logger.info(f"已写出: {all_csv}  ({len(df)} 条)")

    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        out = split_dir / f"{name}.csv"
        sub.to_csv(out, index=False)
        logger.info(f"  {name:5s}: {len(sub):6d} 个切片 → {out.name}")


def print_summary(df: pd.DataFrame):
    logger.info("=" * 55)
    logger.info("重划分统计报告")
    logger.info("=" * 55)
    scenes = df.groupby("split")["scene"].nunique()
    counts = df["split"].value_counts()
    for name in ("train", "val", "test"):
        n_s = scenes.get(name, 0)
        n_p = counts.get(name, 0)
        logger.info(f"  {name:5s}: {n_p:6d} 切片 / {n_s:3d} 场景")
    logger.info(f"  {'合计':5s}: {len(df):6d} 切片 / {df['scene'].nunique():3d} 场景")
    logger.info("=" * 55)


def parse_args():
    parser = argparse.ArgumentParser(description="对已处理切片重新划分 train/val/test")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--seed",   type=int, default=None,
                        help="随机种子（默认读取 config.yaml 中的 random_seed）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计，不写磁盘")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    cfg = get_config(args.config)
    dp  = cfg.data_prep
    paths = cfg.paths

    img_dir   = Path(paths.patch_image_dir)
    msk_dir   = Path(paths.patch_mask_dir)
    split_dir = Path(paths.split_dir)

    train_ratio = float(dp.split_ratio.train)
    val_ratio   = float(dp.split_ratio.val)
    seed        = args.seed if args.seed is not None else int(dp.random_seed)
    allowed_years = _extract_years_from_raw_dirs(paths.raw_sar_dirs)

    logger.info("=" * 55)
    logger.info("冰山数据集切片重划分")
    logger.info(f"  images : {img_dir}")
    logger.info(f"  masks  : {msk_dir}")
    logger.info(f"  splits : {split_dir}")
    logger.info(f"  年份白名单: {sorted(allowed_years) if allowed_years else '未启用'}")
    logger.info(f"  比例   : train={train_ratio:.0%}  val={val_ratio:.0%}  "
                f"test={1-train_ratio-val_ratio:.0%}")
    logger.info(f"  seed   : {seed}")
    logger.info("=" * 55)

    logger.info("扫描已处理切片 ...")
    records = scan_patches(img_dir, msk_dir, allowed_years=allowed_years)
    if not records:
        logger.error("未找到任何有效切片（image + mask 成对存在），请检查路径。")
        return

    n_scenes = len({r["scene"] for r in records})
    logger.info(f"找到 {len(records)} 个切片，来自 {n_scenes} 个场景")

    df = make_splits(records, train_ratio, val_ratio, seed)
    print_summary(df)
    save_splits(df, split_dir, dry_run=args.dry_run)

    if not args.dry_run:
        logger.info("划分完成。")


if __name__ == "__main__":
    main()
