"""
data_prep/pack_hdf5.py
======================
将 data/processed/images + masks 下的散装 .npy 切片合并为每个 split 一个
HDF5 文件，供后续使用 IcebergHDF5Dataset 高速训练。

打包后文件布局（默认写到 data/processed/hdf5/）：
    train.h5  /  val.h5  /  test.h5

每个 HDF5 文件内部结构：
    /images   float32  (N, H, W)   已归一化 SAR 图像
    /masks    uint16   (N, H, W)   冰山实例 ID 掩膜（0=背景）
    /scene    bytes    (N,)        来源场景名
    /row      int32    (N,)        行偏移
    /col      int32    (N,)        列偏移
    attrs: split, n_patches, patch_h, patch_w, next_idx（续传游标）, completed

断点续传：
    中途中断后再次运行同一命令，会从上次中断处继续，不重写已完成的切片。
    写完整个 split 后设置 completed=True，下次运行自动跳过。

使用方式：
    python data_prep/pack_hdf5.py --config configs/config.yaml
    python data_prep/pack_hdf5.py --config configs/config.yaml --splits train
    python data_prep/pack_hdf5.py --config configs/config.yaml --force
"""

import argparse
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Google Drive FUSE 单文件读取超时后的最大重试次数
DEFAULT_MAX_RETRIES  = 6
DEFAULT_BASE_DELAY_S = 2.0    # 首次重试等待秒数（指数退避：2, 4, 8, 16, 32, 64s）
DEFAULT_FLUSH_EVERY  = 200    # 每写 N 个切片刷新 HDF5（减少中断时丢失的工作量）

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


# ──────────────────────────────────────────────────────────────────
# 带退避重试的文件读取
# ──────────────────────────────────────────────────────────────────

def load_npy_with_retry(
    path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
) -> np.ndarray:
    """
    带指数退避重试的 np.load()。
    Google Drive FUSE 偶发 OSError [Errno 5]，重试几次通常可恢复。
    """
    delay = base_delay_s
    for attempt in range(max_retries):
        try:
            return np.load(path)
        except OSError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"读取失败（attempt {attempt+1}/{max_retries}），"
                f"{delay:.0f}s 后重试: {os.path.basename(path)}  [{e}]"
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)   # 最长等 120s


def flush_h5_with_retry(
    h5_file: h5py.File,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
) -> None:
    """Google Drive FUSE 偶发超时时，重试 flush 以保留续传游标。"""
    delay = base_delay_s
    for attempt in range(max_retries):
        try:
            h5_file.flush()
            return
        except OSError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"HDF5 flush 失败（attempt {attempt+1}/{max_retries}），"
                f"{delay:.0f}s 后重试: [{e}]"
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)


def write_sample_with_retry(
    idx: int,
    ds_img,
    ds_msk,
    ds_scene,
    ds_row,
    ds_col,
    img: np.ndarray,
    msk: np.ndarray,
    scene: str,
    row_offset: int,
    col_offset: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
) -> None:
    """单样本写入重试，减少 Google Drive 短暂抖动导致的整批中断。"""
    delay = base_delay_s
    for attempt in range(max_retries):
        try:
            ds_img[idx] = img
            ds_msk[idx] = msk
            ds_scene[idx] = scene
            ds_row[idx] = row_offset
            ds_col[idx] = col_offset
            return
        except OSError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"写入失败（idx={idx}, attempt {attempt+1}/{max_retries}），"
                f"{delay:.0f}s 后重试: [{e}]"
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)


def is_google_drive_path(path: Path) -> bool:
    """粗略判断路径是否位于 Google Drive 挂载目录。"""
    p = str(path).replace("\\", "/").lower()
    return p.startswith("/content/drive") or "/content/drive/" in p


def infer_patch_shape_from_df(
    df: pd.DataFrame,
    max_probe: int = 32,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    prefer_shape: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    从 CSV 中探测实际切片尺寸。
    会统计前 max_probe 条样本的尺寸分布，避免“仅首条样本”导致的误判。
    """
    shape_counts: dict[tuple[int, int], int] = {}
    n_probe = min(len(df), max_probe)
    for i in range(n_probe):
        p = str(df.iloc[i]["image_path"])
        try:
            arr = load_npy_with_retry(
                p,
                max_retries=max_retries,
                base_delay_s=base_delay_s,
            )
            shape = (int(arr.shape[-2]), int(arr.shape[-1]))
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        except Exception as e:
            logger.warning(f"探测切片尺寸失败（probe idx={i}）: {os.path.basename(p)} [{e}]")

    if not shape_counts:
        return None

    if len(shape_counts) > 1:
        detail = ", ".join(
            f"{h}x{w}:{c}" for (h, w), c in sorted(shape_counts.items())
        )
        logger.warning(f"检测到多种切片尺寸（仅统计前 {n_probe} 条）: {detail}")

    if prefer_shape is not None and prefer_shape in shape_counts:
        target = prefer_shape
        logger.info(
            f"探测到配置尺寸 {target[0]}x{target[1]} 出现在样本中，优先使用该尺寸"
        )
        return target

    # 先按出现次数选众数，次数相同则选像素面积更大的尺寸
    target, cnt = max(
        shape_counts.items(),
        key=lambda kv: (kv[1], kv[0][0] * kv[0][1], kv[0][0], kv[0][1]),
    )
    logger.info(f"探测尺寸结果: {target[0]}x{target[1]}（出现 {cnt}/{n_probe}）")
    return target


def filter_df_by_allowed_years(df: pd.DataFrame, allowed_years: set[str] | None) -> pd.DataFrame:
    """仅保留 scene 名称年份出现在白名单中的切片记录。"""
    if not allowed_years:
        return df

    if "scene" not in df.columns:
        logger.warning("CSV 缺少 scene 列，无法按年份白名单过滤，保留原始数据。")
        return df

    scene_years = df["scene"].astype(str).map(_extract_year_from_scene)
    keep_mask = scene_years.isin(allowed_years)
    filtered = df[keep_mask].copy()

    dropped = int((~keep_mask).sum())
    if dropped:
        logger.info(
            f"按 raw_sar_dirs 年份白名单过滤切片: 保留 {len(filtered)}/{len(df)}，"
            f"丢弃 {dropped} 条"
        )
    else:
        logger.info(f"按 raw_sar_dirs 年份白名单过滤切片: 全部 {len(df)} 条均保留")
    return filtered


def pad_or_crop_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """将 2D 数组居中裁剪/填充到目标尺寸，保持 dtype 不变。"""
    src_h, src_w = int(arr.shape[-2]), int(arr.shape[-1])
    if src_h == target_h and src_w == target_w:
        return arr

    # 先居中裁剪到不超过目标尺寸
    crop_top = max((src_h - target_h) // 2, 0)
    crop_left = max((src_w - target_w) // 2, 0)
    crop_bottom = crop_top + min(src_h, target_h)
    crop_right = crop_left + min(src_w, target_w)
    cropped = arr[crop_top:crop_bottom, crop_left:crop_right]

    # 再居中填充到目标尺寸
    out = np.zeros((target_h, target_w), dtype=arr.dtype)
    h, w = int(cropped.shape[-2]), int(cropped.shape[-1])
    out_top = (target_h - h) // 2
    out_left = (target_w - w) // 2
    out[out_top:out_top + h, out_left:out_left + w] = cropped
    return out


# ──────────────────────────────────────────────────────────────────
# HDF5 状态检查
# ──────────────────────────────────────────────────────────────────

def get_h5_state(h5_path: Path) -> dict:
    """
    读取已有 HDF5 文件的状态，用于断点续传。

        Returns:
                {
                    "completed": bool,
                    "next_idx": int,
                    "n_patches": int,
                    "patch_h": int,
                    "patch_w": int,
                }
                注意：patch_h/patch_w 与 n_patches 以数据集真实 shape 为准，
                避免 attrs 过期导致的误判。
                文件不存在时返回 patch_h/patch_w=-1
    """
    if not h5_path.exists():
        return {
            "completed": False,
            "next_idx": 0,
            "n_patches": 0,
            "patch_h": -1,
            "patch_w": -1,
        }
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "images" in f:
                ds_shape = f["images"].shape
                ds_n = int(ds_shape[0])
                ds_h = int(ds_shape[1])
                ds_w = int(ds_shape[2])
            else:
                ds_n = int(f.attrs.get("n_patches", 0))
                ds_h = int(f.attrs.get("patch_h", -1))
                ds_w = int(f.attrs.get("patch_w", -1))

            attr_n = int(f.attrs.get("n_patches", ds_n))
            attr_h = int(f.attrs.get("patch_h", ds_h))
            attr_w = int(f.attrs.get("patch_w", ds_w))
            if attr_n != ds_n or attr_h != ds_h or attr_w != ds_w:
                logger.warning(
                    "检测到 HDF5 attrs 与数据集 shape 不一致，"
                    f"将以数据集实际尺寸为准: attrs=({attr_n},{attr_h},{attr_w}) "
                    f"vs ds=({ds_n},{ds_h},{ds_w})"
                )

            return {
                "completed": bool(f.attrs.get("completed", False)),
                "next_idx":  int(f.attrs.get("next_idx",  0)),
                "n_patches": ds_n,
                "patch_h": ds_h,
                "patch_w": ds_w,
            }
    except Exception as e:
        logger.warning(f"读取 HDF5 状态失败，将视为全新文件: {e}")
        return {
            "completed": False,
            "next_idx": 0,
            "n_patches": 0,
            "patch_h": -1,
            "patch_w": -1,
        }


# ──────────────────────────────────────────────────────────────────
# 核心打包函数
# ──────────────────────────────────────────────────────────────────

def pack_split(
    df: pd.DataFrame,
    split_name: str,
    out_path: Path,
    patch_size: int,
    compression: str = "gzip",
    compression_opts: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    shape_mismatch_policy: str = "adjust",
    allow_overwrite: bool = False,
) -> bool:
    """
    将一个 split 的所有切片写入单个 HDF5 文件，支持断点续传。

    Returns:
        True = 本次调用后文件已完整，False = 出错
    """
    n = len(df)
    state = get_h5_state(out_path)

    # ── 探测实际切片尺寸；若已有 HDF5，则也读取旧尺寸用于一致性检查 ──
    inferred_shape = infer_patch_shape_from_df(
        df,
        max_probe=32,
        max_retries=max_retries,
        base_delay_s=base_delay_s,
        prefer_shape=(patch_size, patch_size),
    )
    if inferred_shape is not None:
        patch_h, patch_w = inferred_shape
    elif state["patch_h"] > 0 and state["patch_w"] > 0:
        patch_h, patch_w = state["patch_h"], state["patch_w"]
        logger.warning(
            f"[{split_name}] 未能探测到切片尺寸，临时使用已有 HDF5 尺寸: "
            f"{patch_h}x{patch_w}"
        )
    else:
        # 新建文件时若无法探测尺寸，直接报错，避免创建错误尺寸导致后续广播异常。
        raise RuntimeError(
            f"[{split_name}] 无法从前 32 个样本探测切片尺寸，请检查 Drive 挂载稳定性后重试"
        )

    # ── 诊断：打印已有文件的状态 ──
    if out_path.exists():
        logger.info(
            f"[{split_name}] 发现已有文件: {out_path}\n"
            f"           n_patches={state['n_patches']}  next_idx={state['next_idx']}  "
            f"completed={state['completed']}  "
            f"shape={state['patch_h']}x{state['patch_w']}"
        )
    else:
        logger.info(f"[{split_name}] 未找到已有文件: {out_path}")

    # ── 判断是否需要（重新）创建文件，并记录具体原因 ──
    shape_mismatch = (
        out_path.exists()
        and state["patch_h"] > 0 and state["patch_w"] > 0
        and (state["patch_h"] != patch_h or state["patch_w"] != patch_w)
    )
    create_reasons = []
    if not out_path.exists():
        create_reasons.append(f"文件不存在: {out_path}")
    if out_path.exists() and state["n_patches"] != n:
        create_reasons.append(
            f"CSV 行数({n}) ≠ HDF5 记录数({state['n_patches']})  "
            f"——可能原因：prepare_dataset.py 重新运行后 CSV 已更新"
        )
    if shape_mismatch:
        create_reasons.append(
            f"切片尺寸变化: {state['patch_h']}x{state['patch_w']} → {patch_h}x{patch_w}"
        )

    need_create = bool(create_reasons)

    if need_create:
        # ── 续传保护：有进度时必须显式 --force 才允许覆盖 ──
        has_progress = out_path.exists() and state["next_idx"] > 0
        if has_progress and not allow_overwrite:
            progress_pct = state["next_idx"] / max(state["n_patches"], 1) * 100
            logger.error(
                f"\n{'='*60}\n"
                f"[{split_name}] 续传保护：检测到已有 {state['next_idx']}/{state['n_patches']}"
                f"（{progress_pct:.1f}%）的进度将被覆盖！\n"
                f"  文件路径 : {out_path}\n"
                f"  重建原因 : {'; '.join(create_reasons)}\n\n"
                f"  常见修复方法：\n"
                f"    1. CSV 行数变化 → 检查是否重新运行了 prepare_dataset.py，\n"
                f"       若是，则原进度无法续传，需要 --force 重新打包。\n"
                f"    2. 文件不在此路径 → 用 --out_dir 指向正确目录后重试。\n"
                f"    3. 确认重建 → 添加 --force 参数。\n"
                f"{'='*60}"
            )
            raise RuntimeError(
                f"[{split_name}] 续传保护触发，已终止。请阅读上方错误信息后决策。"
            )
        for reason in create_reasons:
            logger.warning(f"[{split_name}] 新建原因: {reason}")

    comp_kw = {"compression": compression}
    if compression == "gzip":
        comp_kw["compression_opts"] = compression_opts

    # ── 新建文件（或重建）──
    if need_create:
        logger.info(f"[{split_name}] 新建 HDF5: {out_path}  ({n} 切片, {patch_h}×{patch_w})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dt_str = h5py.string_dtype(encoding="utf-8")
        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("images", shape=(n, patch_h, patch_w), dtype="float32",
                             chunks=(1, patch_h, patch_w), **comp_kw)
            f.create_dataset("masks",  shape=(n, patch_h, patch_w), dtype="uint16",
                             chunks=(1, patch_h, patch_w), **comp_kw)
            f.create_dataset("scene",  shape=(n,), dtype=dt_str)
            f.create_dataset("row",    shape=(n,), dtype="int32")
            f.create_dataset("col",    shape=(n,), dtype="int32")
            f.attrs["split"]      = split_name
            f.attrs["n_patches"]  = n
            f.attrs["patch_h"]    = patch_h
            f.attrs["patch_w"]    = patch_w
            f.attrs["next_idx"]   = 0
            f.attrs["completed"]  = False
        start_idx = 0
    else:
        start_idx = state["next_idx"]
        if start_idx > 0:
            logger.info(
                f"[{split_name}] 续传: 已完成 {start_idx}/{n}，"
                f"从 idx={start_idx} 继续"
            )

    if start_idx >= n:
        logger.info(f"[{split_name}] 已全部写完（续传检查）")
        return True

    # ── 逐条写入，每 FLUSH_EVERY 条刷新游标 ──
    errors = 0
    n_shape_adjusted = 0
    pbar = tqdm(
        range(start_idx, n),
        initial=start_idx, total=n,
        desc=f"  {split_name:5s}", unit="patch", ncols=95,
    )

    with h5py.File(str(out_path), "a") as f:
        ds_img   = f["images"]
        ds_msk   = f["masks"]
        ds_scene = f["scene"]
        ds_row   = f["row"]
        ds_col   = f["col"]
        ds_h, ds_w = int(ds_img.shape[1]), int(ds_img.shape[2])

        for i in pbar:
            row = df.iloc[i]
            try:
                img = load_npy_with_retry(
                    row["image_path"],
                    max_retries=max_retries,
                    base_delay_s=base_delay_s,
                )
                msk = load_npy_with_retry(
                    row["mask_path"],
                    max_retries=max_retries,
                    base_delay_s=base_delay_s,
                )
            except OSError as e:
                errors += 1
                logger.error(f"  [idx={i}] 重试耗尽，写零占位: {e}")
                img = np.zeros((patch_h, patch_w), dtype="float32")
                msk = np.zeros((patch_h, patch_w), dtype="uint16")

            if img.shape[-2:] != (ds_h, ds_w) or msk.shape[-2:] != (ds_h, ds_w):
                if shape_mismatch_policy == "error":
                    raise ValueError(
                        f"[{split_name}] idx={i} 切片尺寸不一致: "
                        f"image={img.shape[-2:]}, mask={msk.shape[-2:]}, h5={(ds_h, ds_w)}"
                    )

                old_img_shape = img.shape[-2:]
                old_msk_shape = msk.shape[-2:]
                img = pad_or_crop_to_shape(img, ds_h, ds_w)
                msk = pad_or_crop_to_shape(msk, ds_h, ds_w)
                n_shape_adjusted += 1
                if n_shape_adjusted <= 5:
                    logger.warning(
                        f"[{split_name}] idx={i} 尺寸不一致，已自动调整: "
                        f"image {old_img_shape}->{img.shape[-2:]}, "
                        f"mask {old_msk_shape}->{msk.shape[-2:]}"
                    )

            write_sample_with_retry(
                idx=i,
                ds_img=ds_img,
                ds_msk=ds_msk,
                ds_scene=ds_scene,
                ds_row=ds_row,
                ds_col=ds_col,
                img=img,
                msk=msk,
                scene=str(row.get("scene", "")),
                row_offset=int(row.get("row_offset", -1)),
                col_offset=int(row.get("col_offset", -1)),
                max_retries=max_retries,
                base_delay_s=base_delay_s,
            )

            # 每 flush_every 条更新续传游标
            if (i - start_idx + 1) % flush_every == 0:
                f.attrs["next_idx"] = i + 1
                flush_h5_with_retry(
                    f,
                    max_retries=max_retries,
                    base_delay_s=base_delay_s,
                )

        # 写完后标记完成
        f.attrs["next_idx"]  = n
        f.attrs["n_errors"]  = errors
        f.attrs["n_shape_adjusted"] = n_shape_adjusted
        f.attrs["completed"] = True
        flush_h5_with_retry(
            f,
            max_retries=max_retries,
            base_delay_s=base_delay_s,
        )

    if errors:
        logger.warning(f"[{split_name}] 完成，{errors} 个切片读取失败（已零填充）")
    else:
        logger.info(f"[{split_name}] 完成，无错误")
    if n_shape_adjusted:
        logger.warning(
            f"[{split_name}] 有 {n_shape_adjusted} 个切片尺寸不一致，已按策略自动调整"
        )

    size_gb = out_path.stat().st_size / 1e9
    logger.info(f"[{split_name}] 文件大小: {size_gb:.2f} GB")
    return True


# ──────────────────────────────────────────────────────────────────
# 并行打包（多线程读 + 单线程写，适合 Drive FUSE 场景）
# ──────────────────────────────────────────────────────────────────

def pack_split_parallel(
    df: pd.DataFrame,
    split_name: str,
    out_path: Path,
    patch_size: int,
    n_workers: int = 32,
    compression: str = "gzip",
    compression_opts: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    shape_mismatch_policy: str = "adjust",
) -> bool:
    """
    并行读取版：n_workers 个线程同时从 Drive 读取 .npy 文件，
    单个写入线程持有 h5py 文件句柄（保证线程安全）。

    与 pack_split（串行）的区别：
      - 无断点续传：中断后需重跑整个 split（但速度快，通常 15-30 分钟）
      - 速度提升 10-30 倍：Drive FUSE 读延迟可被多线程并行掩盖
    """
    n = len(df)

    inferred = infer_patch_shape_from_df(
        df, max_probe=32,
        max_retries=max_retries, base_delay_s=base_delay_s,
        prefer_shape=(patch_size, patch_size),
    )
    if inferred is None:
        raise RuntimeError(f"[{split_name}] 无法探测切片尺寸，请检查 Drive 挂载稳定性")
    patch_h, patch_w = inferred

    comp_kw: dict = {"compression": compression}
    if compression == "gzip":
        comp_kw["compression_opts"] = compression_opts

    logger.info(
        f"[{split_name}] 创建 HDF5（并行 {n_workers} 线程）: "
        f"{out_path}  ({n} 切片, {patch_h}×{patch_w})"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("images", shape=(n, patch_h, patch_w), dtype="float32",
                         chunks=(1, patch_h, patch_w), **comp_kw)
        f.create_dataset("masks",  shape=(n, patch_h, patch_w), dtype="uint16",
                         chunks=(1, patch_h, patch_w), **comp_kw)
        f.create_dataset("scene",  shape=(n,), dtype=dt_str)
        f.create_dataset("row",    shape=(n,), dtype="int32")
        f.create_dataset("col",    shape=(n,), dtype="int32")
        f.attrs.update(split=split_name, n_patches=n,
                       patch_h=patch_h, patch_w=patch_w, completed=False)

    # ── 读取函数（在工作线程中运行）──
    def read_pair(i: int):
        row = df.iloc[i]
        try:
            img = load_npy_with_retry(str(row["image_path"]), max_retries, base_delay_s)
            msk = load_npy_with_retry(str(row["mask_path"]),  max_retries, base_delay_s)
        except OSError as e:
            logger.error(f"  [idx={i}] 读取失败，写零占位: {e}")
            img = np.zeros((patch_h, patch_w), dtype="float32")
            msk = np.zeros((patch_h, patch_w), dtype="uint16")

        def _fix(arr):
            arr2d = arr.squeeze() if arr.ndim == 3 else arr
            return pad_or_crop_to_shape(arr2d, patch_h, patch_w) \
                   if arr2d.shape != (patch_h, patch_w) else arr2d

        return (
            i, _fix(img), _fix(msk),
            str(row.get("scene", "")),
            int(row.get("row_offset", -1)),
            int(row.get("col_offset", -1)),
        )

    # ── 写入线程（单线程，持有唯一 h5py 句柄）──
    _SENTINEL = object()
    write_q   = Queue(maxsize=n_workers * 3)
    n_written = [0]
    n_errors  = [0]

    def writer():
        with h5py.File(str(out_path), "a") as f:
            ds_img, ds_msk = f["images"], f["masks"]
            ds_scene, ds_row, ds_col = f["scene"], f["row"], f["col"]
            cnt = 0
            while True:
                item = write_q.get()
                if item is _SENTINEL:
                    break
                idx, img, msk, scene, r_off, c_off = item
                try:
                    ds_img[idx]   = img
                    ds_msk[idx]   = msk
                    ds_scene[idx] = scene
                    ds_row[idx]   = r_off
                    ds_col[idx]   = c_off
                    n_written[0] += 1
                except Exception as e:
                    n_errors[0] += 1
                    logger.error(f"  [idx={idx}] 写入失败: {e}")
                cnt += 1
                if cnt % 500 == 0:
                    f.flush()
            f.attrs["completed"] = True
            f.attrs["n_written"] = n_written[0]
            f.flush()

    wt = Thread(target=writer, daemon=True)
    wt.start()

    # ── 并行读取（滑动窗口）+ 投入写入队列 ──
    #
    # 关键设计：不一次性提交全部任务。
    # 若将 n=60000 个 Future 全部提交，每个 Future 的结果（~1.5MB）在
    # fut.result() 被调用后依然保留在 Future 对象里，直到 futures dict 销毁，
    # 极端情况下会占用 60000 × 1.5MB ≈ 90GB RAM。
    #
    # 滑动窗口做法：最多保持 max_in_flight 个 Future 在内存，
    # 处理完一个立即 del 释放，内存稳定在约 max_in_flight × 1.5MB。
    from concurrent.futures import wait, FIRST_COMPLETED

    max_in_flight = n_workers * 4          # 128 个 Future，约 192MB
    pbar = tqdm(total=n, desc=f"  {split_name:5s}", unit="patch", ncols=95)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        in_flight: dict = {}               # {Future: index}
        submit_idx = 0

        while submit_idx < n or in_flight:
            # 补充任务，直到达到 max_in_flight 上限
            while submit_idx < n and len(in_flight) < max_in_flight:
                fut = pool.submit(read_pair, submit_idx)
                in_flight[fut] = submit_idx
                submit_idx += 1

            if not in_flight:
                break

            # 等待至少一个完成
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                write_q.put(fut.result())
                del in_flight[fut]         # 立即释放 Future 及其结果
                pbar.update(1)

    pbar.close()

    write_q.put(_SENTINEL)
    wt.join()

    if n_errors[0]:
        logger.warning(f"[{split_name}] 完成，{n_errors[0]} 个切片写入失败")
    else:
        logger.info(f"[{split_name}] 完成，写入 {n_written[0]}/{n} 个切片")
    logger.info(f"[{split_name}] 文件大小: {out_path.stat().st_size / 1e9:.2f} GB")
    return True


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="将散装 .npy 切片打包为 HDF5（支持断点续传）")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="HDF5 输出目录（默认: <processed_dir>/hdf5）",
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf"],
    )
    parser.add_argument("--gzip_level", type=int, default=1)
    parser.add_argument(
        "--force", action="store_true",
        help="忽略 completed 标记，强制重新打包",
    )
    parser.add_argument(
        "--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="I/O 超时重试次数（默认: 6）",
    )
    parser.add_argument(
        "--base_delay_s", type=float, default=DEFAULT_BASE_DELAY_S,
        help="首次重试等待秒数（指数退避，默认: 2.0）",
    )
    parser.add_argument(
        "--flush_every", type=int, default=DEFAULT_FLUSH_EVERY,
        help="每 N 个切片刷新一次 HDF5 续传游标（默认: 200）",
    )
    parser.add_argument(
        "--shape_mismatch_policy",
        default="adjust",
        choices=["adjust", "error"],
        help="切片尺寸不一致策略：adjust=自动居中裁剪/填充，error=直接报错",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "并行读取线程数（默认 1 = 串行模式，支持断点续传）。"
            "设为 32 可开启并行模式：速度提升 10-30 倍，但无断点续传。"
            "推荐在 Drive FUSE 环境下使用 --workers 32 --local_tmp /content/hdf5_tmp"
        ),
    )
    parser.add_argument(
        "--local_tmp", default=None,
        help=(
            "本地临时写入目录（如 /content/hdf5_tmp）。"
            "HDF5 先写到本地 SSD（写入快），完成后自动拷回 out_dir（Drive）。"
            "与 --workers 配合使用效果最佳。"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    cfg   = get_config(args.config)
    paths = cfg.paths

    if args.max_retries < 1:
        raise ValueError("--max_retries 必须 >= 1")
    if args.base_delay_s <= 0:
        raise ValueError("--base_delay_s 必须 > 0")
    if args.flush_every < 1:
        raise ValueError("--flush_every 必须 >= 1")

    split_dir  = Path(paths.split_dir)
    out_dir    = Path(args.out_dir) if args.out_dir else Path(paths.processed_dir) / "hdf5"
    patch_size = int(cfg.data_prep.patch_size)
    allowed_years = _extract_years_from_raw_dirs(paths.raw_sar_dirs)

    out_dir.mkdir(parents=True, exist_ok=True)

    use_parallel = (args.workers > 1)
    local_tmp    = Path(args.local_tmp) if args.local_tmp else None

    logger.info("=" * 60)
    logger.info(f"HDF5 数据打包（{'并行 ' + str(args.workers) + ' 线程' if use_parallel else '串行+断点续传'}）")
    logger.info(f"  split_dir  : {split_dir}")
    logger.info(f"  out_dir    : {out_dir}")
    if local_tmp:
        logger.info(f"  local_tmp  : {local_tmp}  （写完后拷回 out_dir）")
    logger.info(f"  splits     : {args.splits}")
    logger.info(f"  年份白名单 : {sorted(allowed_years) if allowed_years else '未启用'}")
    logger.info(f"  patch_size : {patch_size}x{patch_size}")
    logger.info(f"  压缩       : {args.compression}" +
                (f"  level={args.gzip_level}" if args.compression == "gzip" else ""))
    if use_parallel:
        logger.info(f"  并行线程数 : {args.workers}（无断点续传）")
    else:
        logger.info(
            f"  重试策略   : max_retries={args.max_retries}, "
            f"base_delay_s={args.base_delay_s}"
        )
        logger.info(f"  续传粒度   : 每 {args.flush_every} 个切片刷新一次游标")
    logger.info(f"  尺寸策略   : {args.shape_mismatch_policy}")
    logger.info("=" * 60)

    total_start = time.time()

    for split_name in args.splits:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            logger.warning(f"找不到 {csv_path}，跳过 {split_name}")
            continue

        # 并行模式写到本地临时目录（若指定），完成后再拷回 Drive
        final_out_path = out_dir / f"{split_name}.h5"
        write_path     = (local_tmp / f"{split_name}.h5") if local_tmp else final_out_path

        # ── completed 检查（并行模式也检查 final_out_path）──
        state = get_h5_state(final_out_path)
        if not args.force and state["completed"]:
            shape_now = None
            try:
                logger.info(f"[{split_name}] 已标记 completed，进行尺寸一致性检查...")
                df_probe  = pd.read_csv(csv_path)
                shape_now = infer_patch_shape_from_df(
                    df_probe, max_probe=16,
                    max_retries=args.max_retries, base_delay_s=args.base_delay_s,
                    prefer_shape=(patch_size, patch_size),
                )
            except Exception as e:
                logger.warning(f"[{split_name}] completed 校验失败，按原逻辑跳过: {e}")

            if (
                shape_now is not None
                and state["patch_h"] > 0 and state["patch_w"] > 0
                and (shape_now[0] != state["patch_h"] or shape_now[1] != state["patch_w"])
            ):
                logger.warning(
                    f"[{split_name}] completed 文件尺寸过期："
                    f"h5={state['patch_h']}x{state['patch_w']} vs "
                    f"csv={shape_now[0]}x{shape_now[1]}，将重建"
                )
            else:
                logger.info(f"[{split_name}] {final_out_path.name} 已完整，跳过（--force 可强制重写）")
                continue

        logger.info(f"\n读取 {csv_path.name} ...")
        df = pd.read_csv(csv_path)
        df = filter_df_by_allowed_years(df, allowed_years)

        if df.empty:
            logger.warning(f"[{split_name}] 过滤后无可打包样本，跳过")
            continue

        t0 = time.time()

        if use_parallel:
            if local_tmp:
                local_tmp.mkdir(parents=True, exist_ok=True)
            pack_split_parallel(
                df=df,
                split_name=split_name,
                out_path=write_path,
                patch_size=patch_size,
                n_workers=args.workers,
                compression=args.compression,
                compression_opts=args.gzip_level,
                max_retries=args.max_retries,
                base_delay_s=args.base_delay_s,
                shape_mismatch_policy=args.shape_mismatch_policy,
            )
            # 写到本地后拷回 Drive
            if local_tmp and write_path != final_out_path:
                logger.info(f"[{split_name}] 拷回 Drive: {write_path} → {final_out_path}")
                final_out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(write_path), str(final_out_path))
                write_path.unlink()
                logger.info(f"[{split_name}] 拷贝完成，本地临时文件已删除")
        else:
            pack_split(
                df=df,
                split_name=split_name,
                out_path=write_path,
                patch_size=patch_size,
                compression=args.compression,
                compression_opts=args.gzip_level,
                max_retries=args.max_retries,
                base_delay_s=args.base_delay_s,
                flush_every=args.flush_every,
                shape_mismatch_policy=args.shape_mismatch_policy,
                allow_overwrite=args.force,
            )

        elapsed = time.time() - t0
        logger.info(f"[{split_name}] 耗时 {elapsed / 60:.1f} 分钟")

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部完成，总耗时 {total_elapsed / 60:.1f} 分钟")
    logger.info(f"HDF5 文件位于: {out_dir}")


if __name__ == "__main__":
    main()
