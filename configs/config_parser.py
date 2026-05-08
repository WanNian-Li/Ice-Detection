"""
configs/config_parser.py
========================
配置文件解析工具：读取 config.yaml，返回一个支持"点访问"的配置对象。
使用方式:
    from configs.config_parser import get_config
    cfg = get_config("configs/config.yaml")
    print(cfg.train.epochs)
    print(cfg.paths.checkpoint_dir)
"""

import os
from pathlib import Path

import yaml
from omegaconf import OmegaConf, DictConfig


def get_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """
    加载并返回 OmegaConf 配置对象。

    Args:
        config_path: config.yaml 的路径（相对或绝对均可）

    Returns:
        OmegaConf DictConfig 对象，支持 cfg.key.subkey 点式访问
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.resolve()}")

    # 使用 OmegaConf 加载，自动支持类型推断和嵌套结构
    cfg = OmegaConf.load(config_path)

    # 将所有相对路径解析为以 project_root 为基准的绝对路径
    cfg = _resolve_paths(cfg)

    return cfg


def _resolve_paths(cfg: DictConfig) -> DictConfig:
    """
    将 paths 节点下所有相对路径转换为绝对路径，
    并自动创建不存在的输出目录。
    """
    root = Path(cfg.project_root)

    # 需要自动创建的目录键
    auto_create_keys = {
        "checkpoint_dir", "log_dir", "prediction_dir",
        "patch_image_dir", "patch_mask_dir", "split_dir",
    }

    from omegaconf import ListConfig
    for key, value in cfg.paths.items():
        # 列表类型（如 raw_sar_dirs）：逐项解析相对路径，跳过绝对路径
        if isinstance(value, ListConfig):
            resolved = [
                str(root / v) if not Path(v).is_absolute() else str(v)
                for v in value
            ]
            OmegaConf.update(cfg, f"paths.{key}", resolved)
            continue

        abs_path = root / value
        OmegaConf.update(cfg, f"paths.{key}", str(abs_path))

        if key in auto_create_keys:
            abs_path.mkdir(parents=True, exist_ok=True)

    return cfg


def save_config(cfg: DictConfig, save_path: str) -> None:
    """
    将当前配置序列化保存到指定路径（便于实验可复现）。

    Args:
        cfg:       OmegaConf 配置对象
        save_path: 保存路径，例如 "outputs/logs/run_01/config.yaml"
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path)
    print(f"[Config] 配置已保存到: {save_path}")


if __name__ == "__main__":
    # 快速验证：直接运行此脚本检查配置是否能正确加载
    cfg = get_config("configs/config.yaml")
    print(OmegaConf.to_yaml(cfg))
