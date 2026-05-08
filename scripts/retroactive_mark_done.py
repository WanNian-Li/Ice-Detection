"""
scripts/retroactive_mark_done.py
==================================
一个一次性回填脚本，用于为已经存在的切片数据创建场景完成标记。

背景：
在 data_prep/prepare_dataset.py 中引入了场景级断点续传功能，
该功能依赖于 `outputs/processed/splits/scene_done/` 目录下的
`.done.json` 文件来判断一个 SAR 场景是否已处理完成。

对于在该功能引入前就已经处理好的旧数据，它们缺少这些标记文件。
本脚本的作用就是扫描已有的 masks 目录，根据 .npy 文件名反推出
所有已处理的场景，并为它们补上 .done.json 标记。

使用方式：
    python scripts/retroactive_mark_done.py
"""

import json
import os
import re
import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_parser import get_config

def main():
    """脚本主入口"""
    print("=" * 60)
    print("开始回填历史场景的完成标记...")
    print("=" * 60)

    # --- 加载配置 ---
    # 切换到项目根目录，使相对路径正常工作
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        cfg = get_config("configs/config.yaml")
    except FileNotFoundError:
        print("错误：无法找到 configs/config.yaml 文件。")
        print("请确保您在 Iceberg-Detection 项目的根目录下运行此脚本。")
        return

    paths = cfg.paths
    patch_mask_dir = Path(paths.patch_mask_dir)
    scene_done_dir = Path(paths.split_dir) / "scene_done"

    if not patch_mask_dir.exists():
        print(f"错误：掩膜目录不存在，无法扫描: {patch_mask_dir}")
        return

    # --- 确保目标目录存在 ---
    scene_done_dir.mkdir(parents=True, exist_ok=True)
    print(f"标记文件将被写入: {scene_done_dir}")

    # --- 扫描并提取场景名 ---
    # 文件名格式: {scene_name}_r{row:05d}_c{col:05d}.npy
    # 我们需要用正则表达式把 scene_name 提取出来
    pattern = re.compile(r"^(.*?)_r\d{5,}_c\d{5,}\.npy$")
    
    processed_scenes = set()
    print(f"\n正在扫描 {patch_mask_dir} ...")
    
    for mask_file in patch_mask_dir.glob("*.npy"):
        match = pattern.match(mask_file.name)
        if match:
            scene_name = match.group(1)
            processed_scenes.add(scene_name)

    if not processed_scenes:
        print("\n在掩膜目录中未找到任何已处理的切片文件。无需回填。")
        return

    print(f"扫描完成，共找到 {len(processed_scenes)} 个已处理的独立场景。")

    # --- 回填创建标记文件 ---
    created_count = 0
    skipped_count = 0

    for scene_name in sorted(list(processed_scenes)):
        done_file = scene_done_dir / f"{scene_name}.done.json"

        if done_file.exists():
            # 如果标记已存在，则跳过
            skipped_count += 1
            continue

        # 根据要求，kept_patches 设为 -1 表示是回填的
        payload = {
            "scene": scene_name,
            "kept_patches": -1,
        }

        try:
            done_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [CREATED] {done_file.name}")
            created_count += 1
        except IOError as e:
            print(f"  [ERROR] 无法写入文件 {done_file.name}: {e}")

    print("\n" + "=" * 60)
    print("回填报告:")
    print(f"  成功创建标记: {created_count} 个")
    print(f"  已存在并跳过: {skipped_count} 个")
    print("=" * 60)
    print("操作完成。现在您可以正常运行 prepare_dataset.py 脚本了。")


if __name__ == "__main__":
    main()
