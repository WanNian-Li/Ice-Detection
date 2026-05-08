"""
Analyze patch size distribution from split CSVs.

What this script reports:
1. Ratio of 256x256 and 512x512 patches (overall and per split)
2. Year distribution for 256x256 patches

Usage (run in cloud):
    python scripts/analyze_patch_sizes.py --config configs/config.yaml
    python scripts/analyze_patch_sizes.py --config configs/config.yaml --splits train
    python scripts/analyze_patch_sizes.py --config configs/config.yaml --workers 16
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config


YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?:\d{2})?(?!\d)")


@dataclass
class RowResult:
    split: str
    size: tuple[int, int] | None
    year: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 256/512 patch ratio and years of 256 patches"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="Number of worker threads for reading npy headers",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap per split for quick debug (default: all)",
    )
    return parser.parse_args()


def extract_year(scene: object, image_path: str) -> str:
    """Extract year from scene or image path. Returns 'unknown' if unavailable."""
    candidates: list[str] = []
    if scene is not None and not (isinstance(scene, float) and np.isnan(scene)):
        candidates.append(str(scene))
    candidates.append(image_path)

    for text in candidates:
        m = YEAR_PATTERN.search(text)
        if m:
            return m.group(1)
    return "unknown"


def resolve_image_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return project_root / p


def read_shape(image_path: Path) -> tuple[int, int] | None:
    """Read npy shape without loading full array into memory."""
    arr = np.load(str(image_path), mmap_mode="r")
    if arr.ndim < 2:
        return None
    return int(arr.shape[-2]), int(arr.shape[-1])


def process_rows(
    split_name: str,
    df: pd.DataFrame,
    project_root: Path,
    workers: int,
) -> Iterable[RowResult]:
    if "image_path" not in df.columns:
        raise ValueError(f"[{split_name}] CSV missing required column: image_path")

    scenes = df["scene"] if "scene" in df.columns else pd.Series([None] * len(df))

    tasks: list[tuple[str, str | None]] = []
    for raw_path, scene in zip(df["image_path"].astype(str), scenes):
        tasks.append((raw_path, None if pd.isna(scene) else str(scene)))

    def worker(task: tuple[str, str | None]) -> RowResult:
        raw_path, scene = task
        full_path = resolve_image_path(raw_path, project_root)
        year = extract_year(scene, raw_path)
        try:
            size = read_shape(full_path)
            return RowResult(split=split_name, size=size, year=year, error=None)
        except Exception as e:  # noqa: BLE001
            return RowResult(split=split_name, size=None, year=year, error=str(e))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for result in tqdm(
            ex.map(worker, tasks),
            total=len(tasks),
            desc=f"scan {split_name}",
            unit="patch",
            ncols=100,
        ):
            yield result


def ratio_str(n: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * n / total):.2f}%"


def main() -> None:
    args = parse_args()

    cfg = get_config(args.config)
    project_root = Path(cfg.project_root)
    split_dir = Path(cfg.paths.split_dir)

    global_size_counter: Counter[tuple[int, int]] = Counter()
    global_year_256: Counter[str] = Counter()
    global_errors = 0

    per_split_size: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    per_split_errors: dict[str, int] = defaultdict(int)

    print("=" * 72)
    print("Patch size analysis")
    print(f"split_dir: {split_dir}")
    print(f"splits   : {args.splits}")
    print(f"workers  : {args.workers}")
    if args.max_rows is not None:
        print(f"max_rows : {args.max_rows} per split")
    print("=" * 72)

    for split_name in args.splits:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"[WARN] missing split csv: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if args.max_rows is not None:
            df = df.head(args.max_rows)

        for result in process_rows(
            split_name=split_name,
            df=df,
            project_root=project_root,
            workers=max(1, args.workers),
        ):
            if result.error is not None:
                global_errors += 1
                per_split_errors[split_name] += 1
                continue

            if result.size is None:
                global_errors += 1
                per_split_errors[split_name] += 1
                continue

            global_size_counter[result.size] += 1
            per_split_size[split_name][result.size] += 1

            if result.size == (256, 256):
                global_year_256[result.year] += 1

    n_256 = global_size_counter[(256, 256)]
    n_512 = global_size_counter[(512, 512)]
    n_all = sum(global_size_counter.values())
    n_256_512 = n_256 + n_512

    print()
    print("[Overall]")
    print(f"valid patches           : {n_all}")
    print(f"read/shape errors       : {global_errors}")
    print(f"count 256x256           : {n_256}")
    print(f"count 512x512           : {n_512}")
    print(f"ratio 256 in all        : {ratio_str(n_256, n_all)}")
    print(f"ratio 512 in all        : {ratio_str(n_512, n_all)}")
    print(f"ratio 256 in (256+512)  : {ratio_str(n_256, n_256_512)}")
    print(f"ratio 512 in (256+512)  : {ratio_str(n_512, n_256_512)}")

    print()
    print("[Per split]")
    for split_name in args.splits:
        counter = per_split_size.get(split_name, Counter())
        s256 = counter[(256, 256)]
        s512 = counter[(512, 512)]
        sall = sum(counter.values())
        s256_512 = s256 + s512
        serr = per_split_errors.get(split_name, 0)

        print(f"- {split_name}")
        print(f"  valid patches          : {sall}")
        print(f"  read/shape errors      : {serr}")
        print(f"  count 256x256          : {s256}")
        print(f"  count 512x512          : {s512}")
        print(f"  ratio 256 in all       : {ratio_str(s256, sall)}")
        print(f"  ratio 512 in all       : {ratio_str(s512, sall)}")
        print(f"  ratio 256 in (256+512) : {ratio_str(s256, s256_512)}")
        print(f"  ratio 512 in (256+512) : {ratio_str(s512, s256_512)}")

    print()
    print("[Year distribution for 256x256]")
    if not global_year_256:
        print("No 256x256 patches found.")
    else:
        total_256 = sum(global_year_256.values())
        for year, cnt in sorted(global_year_256.items(), key=lambda x: x[0]):
            print(f"- {year}: {cnt} ({ratio_str(cnt, total_256)})")


if __name__ == "__main__":
    main()
