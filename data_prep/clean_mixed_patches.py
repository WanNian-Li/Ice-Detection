"""
data_prep/clean_mixed_patches.py
=================================
检查 data/processed/images/ 中混杂的不同尺寸切片，
按「尺寸 × 年份」打印分布表，确认后删除指定尺寸的切片（images/ + masks/）。

核心设计：绝大多数文件通过文件名偏移量推断尺寸（零 I/O），
仅对 r=0, c=0 的角点文件（每个场景只有一个）读取 header，
数万个文件可在几秒内完成分类，不受 Drive FUSE 延迟影响。

使用方式：
    python data_prep/clean_mixed_patches.py
    python data_prep/clean_mixed_patches.py --target_size 256
    python data_prep/clean_mixed_patches.py --dry_run   # 只统计，不删除
    python data_prep/clean_mixed_patches.py --yes       # 跳过交互确认
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config


YEAR_RE         = re.compile(r"(20\d{2})")
OFFSET_RE       = re.compile(r"_r(\d+)_c(\d+)$")


# ──────────────────────────────────────────────────────────────────
# 文件枚举（os.listdir，FUSE 下比 glob 可靠）
# ──────────────────────────────────────────────────────────────────

def list_npy_files(directory: Path) -> list:
    """
    用 os.listdir() 枚举 .npy 文件。
    比 Path.glob() 在 Google Drive FUSE 挂载下更可靠：
    glob() 底层依赖 scandir，FUSE 偶发缓存未刷新时会返回空列表；
    listdir() 直接触发 FUSE readdir syscall，强制刷新目录缓存。
    """
    try:
        names = os.listdir(str(directory))
    except OSError as e:
        print(f"[错误] 无法列出目录内容: {e}")
        return []
    return sorted(directory / n for n in names if n.endswith(".npy"))


# ──────────────────────────────────────────────────────────────────
# 尺寸推断
# ──────────────────────────────────────────────────────────────────

def infer_size_by_stride(stem: str, size_to_stride: dict) -> int | None:
    """
    从文件名偏移量推断 patch 尺寸，零 I/O。

    原理：不同 patch_size 使用不同 stride（= patch_size - overlap），
    文件名中的 row/col 偏移必然是各自 stride 的整数倍。
    只要两种 stride 的 LCM 足够大（256/512 对应 stride 192/448，LCM=1344），
    绝大多数偏移值可唯一对应一种 stride。

    返回 patch_size（int）或 None（无法从文件名判断，需读 header）。
    """
    m = OFFSET_RE.search(stem)
    if not m:
        return None

    row, col = int(m.group(1)), int(m.group(2))

    # r=0, c=0 是任何 stride 都合法的角点，无法区分
    if row == 0 and col == 0:
        return None

    # 用非零偏移值判断：哪个 stride 整除它
    ref = row if row > 0 else col
    matches = [sz for sz, st in size_to_stride.items() if ref % st == 0]

    return matches[0] if len(matches) == 1 else None


def read_npy_shape(path: Path) -> tuple | None:
    """只读 .npy 文件头部获取 shape，不加载数组数据。"""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            reader = (
                np.lib.format.read_array_header_2_0
                if version[0] == 2
                else np.lib.format.read_array_header_1_0
            )
            shape, _, _ = reader(f)
        return tuple(int(d) for d in shape)
    except Exception:
        # 备用：mmap
        try:
            arr = np.load(str(path), mmap_mode="r")
            s = tuple(arr.shape)
            del arr
            return s
        except Exception:
            return None


def read_shape_worker(path: Path) -> tuple:
    """ThreadPoolExecutor worker：返回 (path, shape_or_None)。"""
    return path, read_npy_shape(path)


# ──────────────────────────────────────────────────────────────────
# 报表打印
# ──────────────────────────────────────────────────────────────────

def extract_year(stem: str) -> str:
    m = YEAR_RE.search(stem)
    return m.group(1) if m else "unknown"


def print_table(size_year_counts: dict) -> dict:
    """打印「尺寸 × 年份」分布表，返回 {size_tuple: 合计}。"""
    if not size_year_counts:
        print("  （无数据）")
        return {}

    all_sizes = sorted({hw for hw, _ in size_year_counts}, key=lambda x: x[0])
    all_years = sorted({yr for _, yr in size_year_counts})

    col_w = 9
    header_years = "".join(f"{y:>{col_w}}" for y in all_years)
    print(f"\n  {'尺寸':>12}  {header_years}  {'合计':>{col_w}}")
    sep = "─" * (16 + col_w * (len(all_years) + 1) + 2 * len(all_years))
    print(f"  {sep}")

    size_grand = {}
    for hw in all_sizes:
        row_vals = [size_year_counts.get((hw, yr), 0) for yr in all_years]
        total    = sum(row_vals)
        size_grand[hw] = total
        label = f"{hw[0]}×{hw[1]}"
        cols  = "".join(f"{v:>{col_w}}" for v in row_vals)
        print(f"  {label:>12}  {cols}  {total:>{col_w}}")

    print(f"  {sep}")
    col_totals = [
        sum(size_year_counts.get((hw, yr), 0) for hw in all_sizes)
        for yr in all_years
    ]
    grand = sum(col_totals)
    cols  = "".join(f"{v:>{col_w}}" for v in col_totals)
    print(f"  {'合计':>12}  {cols}  {grand:>{col_w}}")

    return size_grand


# ──────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="检查并清理 images/ + masks/ 中混杂尺寸的 .npy 切片（零 I/O 快速模式）"
    )
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--image_dir",   default=None)
    parser.add_argument("--mask_dir",    default=None)
    parser.add_argument(
        "--target_size", type=int, default=256,
        help="要删除的切片边长（默认 256，即删除 256×256 的切片）",
    )
    parser.add_argument(
        "--known_sizes", type=int, nargs="+", default=[256, 512],
        help="数据集中存在的所有 patch 尺寸（默认: 256 512）",
    )
    parser.add_argument("--dry_run", action="store_true", help="只统计，不删除")
    parser.add_argument("--yes",     action="store_true", help="跳过交互确认直接删除")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    cfg = get_config(args.config)

    img_dir  = Path(args.image_dir or cfg.paths.patch_image_dir)
    msk_dir  = Path(args.mask_dir  or cfg.paths.patch_mask_dir)
    target   = args.target_size
    overlap  = int(cfg.data_prep.overlap)

    # stride = patch_size - overlap
    size_to_stride = {sz: sz - overlap for sz in args.known_sizes}
    stride_info    = "  ".join(f"{sz}px→stride{st}" for sz, st in size_to_stride.items())
    print(f"\n已知尺寸与 stride：{stride_info}（overlap={overlap}）")

    if not img_dir.exists():
        print(f"[错误] images 目录不存在: {img_dir}")
        sys.exit(1)

    # ── 枚举文件 ──
    print(f"\n扫描目录: {img_dir}")
    try:
        all_entries = os.listdir(str(img_dir))
        n_npy = sum(1 for n in all_entries if n.endswith(".npy"))
        print(f"目录条目总数: {len(all_entries):,}  其中 .npy: {n_npy:,}")
    except OSError as e:
        print(f"[警告] 目录诊断失败: {e}")

    all_files = list_npy_files(img_dir)
    print(f"共找到 {len(all_files):,} 个切片文件\n")

    if not all_files:
        print("[提示] 未找到任何 .npy 文件。")
        print("       若 Drive 刚重新挂载，请先运行：  !ls \"" + str(img_dir) + "\" | head")
        return

    # ──────────────────────────────────────────
    # 阶段 1：文件名偏移推断（零 I/O，瞬时完成）
    # ──────────────────────────────────────────
    size_year_counts: dict = defaultdict(int)
    to_delete_stems:  list = []
    ambiguous_paths:  list = []   # 需要读 header 才能判断的文件（通常很少）

    print("阶段 1/2：文件名偏移推断（无 I/O）...")
    for path in tqdm(all_files, desc="解析文件名", unit="file", ncols=95, mininterval=0.5):
        inferred = infer_size_by_stride(path.stem, size_to_stride)
        if inferred is not None:
            hw   = (inferred, inferred)
            year = extract_year(path.stem)
            size_year_counts[(hw, year)] += 1
            if inferred == target:
                to_delete_stems.append(path.stem)
        else:
            ambiguous_paths.append(path)

    print(f"  文件名推断完成：{len(all_files) - len(ambiguous_paths):,} 个已分类，"
          f"{len(ambiguous_paths):,} 个需读 header（角点切片）")

    # ──────────────────────────────────────────
    # 阶段 2：并行读取 header（仅处理角点文件）
    # ──────────────────────────────────────────
    if ambiguous_paths:
        workers = min(32, len(ambiguous_paths))
        print(f"\n阶段 2/2：并行读取 {len(ambiguous_paths):,} 个角点文件的 header"
              f"（{workers} 线程）...")
        read_errors = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(read_shape_worker, p): p for p in ambiguous_paths}
            for fut in tqdm(as_completed(futures), total=len(ambiguous_paths),
                            desc="读取 header", unit="file", ncols=95):
                path, shape = fut.result()
                if shape is None:
                    read_errors.append(path.name)
                    continue
                hw   = (shape[-2], shape[-1]) if len(shape) >= 2 else (shape[0], shape[0])
                year = extract_year(path.stem)
                size_year_counts[(hw, year)] += 1
                if hw[0] == target and hw[1] == target:
                    to_delete_stems.append(path.stem)

        if read_errors:
            print(f"\n[警告] {len(read_errors)} 个文件 header 读取失败（已跳过）")

    # ── 打印分布表 ──
    print("\n" + "=" * 60)
    print("  切片尺寸 × 年份 分布")
    print("=" * 60)
    size_grand = print_table(size_year_counts)

    # ── 待删除摘要 ──
    n_delete   = len(to_delete_stems)
    target_str = f"{target}×{target}"
    print("\n" + "=" * 60)

    if n_delete == 0:
        print(f"  未发现任何 {target_str} 切片，无需操作。")
        print("=" * 60)
        return

    total_all = sum(size_grand.values())
    print(f"  待删除：{n_delete:,} 个 {target_str} 切片")
    print(f"          占全部切片的 {n_delete/total_all*100:.1f}%")
    print(f"  目录：  images/ → {img_dir}")
    print(f"          masks/  → {msk_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry_run] 未执行删除操作。")
        return

    # ── 交互确认 ──
    if not args.yes:
        print()
        answer = input(f'输入 "yes" 确认删除所有 {target_str} 切片，其他任意键取消：  ')
        if answer.strip().lower() != "yes":
            print("已取消，未删除任何文件。")
            return

    # ── 执行删除（并行，缓解 FUSE 单次 unlink 延迟）──
    del_workers = min(64, len(to_delete_stems))
    print(f"\n并行删除（{del_workers} 线程，减少 Drive FUSE 延迟影响）...\n")

    from threading import Lock
    counts = {"img": 0, "msk": 0, "miss": 0}
    lock   = Lock()

    def delete_pair(stem: str):
        img_path = img_dir / f"{stem}.npy"
        msk_path = msk_dir / f"{stem}.npy"
        d_img = d_msk = miss = 0
        try:
            img_path.unlink()
            d_img = 1
        except FileNotFoundError:
            pass
        try:
            msk_path.unlink()
            d_msk = 1
        except FileNotFoundError:
            miss = 1
        with lock:
            counts["img"]  += d_img
            counts["msk"]  += d_msk
            counts["miss"] += miss

    with ThreadPoolExecutor(max_workers=del_workers) as pool:
        futures = [pool.submit(delete_pair, stem) for stem in to_delete_stems]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="删除切片", unit="file", ncols=95):
            pass

    print(f"\n删除完成：")
    print(f"  images/ 已删除：{counts['img']:,} 个")
    print(f"  masks/  已删除：{counts['msk']:,} 个")
    if counts["miss"]:
        print(f"  masks/  未找到对应文件（已忽略）：{counts['miss']:,} 个")

    print(f"\n[提示] 请重新运行以下命令重建 split CSV 和 HDF5：")
    print(f"  python data_prep/prepare_dataset.py --config configs/config.yaml")
    print(f"  python data_prep/pack_hdf5.py --config configs/config.yaml --force")


if __name__ == "__main__":
    main()
