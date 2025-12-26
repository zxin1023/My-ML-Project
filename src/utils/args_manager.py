import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import asdict, fields


from configs.base_config import BaseConfig


class ArgsManager:
    """
    管理命令行参数和配置文件参数的类。
    该类负责解析命令行参数、合并配置文件参数和命令行参数、
    并将结果存储在一个属性中，方便后续使用。
    """
     
    @staticmethod   
    def get_parser() -> argparse.ArgumentParser:
        """
        创建并返回一个 ArgumentParser 实例，用于解析命令行参数。
        返回:
            argparse.ArgumentParser: 配置好的 ArgumentParser 实例。
        """
        parser = argparse.ArgumentParser(description="Training and Evaluation Script")
        
        # 添加配置文件路径参数
        parser.add_argument("--config", type=Path, default=None, help="Path to the config file")
        # 添加其他命令行参数
        
        return parser
    
    @staticmethod   
    def parse_args_and_update_config(config_path: str) -> BaseConfig:
        """
        解析命令行参数并更新配置文件参数。
        参数:
            config (BaseConfig): 配置文件对象，包含默认参数。
        返回:
            BaseConfig: 更新后的配置文件对象，包含合并后的参数。
        """
        # 解析命令行参数
        parser = ArgsManager.get_parser()
        args = parser.parse_args()
        # 加载配置文件
        config = BaseConfig.load_config(config_path)
        # 更新配置文件参数
        for field in fields(config):
            field_name = field.name
            if hasattr(args, field_name):
                setattr(config, field_name, getattr(args, field_name))
        return config
    
    @staticmethod   
    def load_config_from_file(config_path: str) -> BaseConfig:
        """
        从文件加载配置参数。
        参数:
            config_path (str): 配置文件路径。
        返回:
            BaseConfig: 加载后的配置文件对象。
        """
        return BaseConfig.load_config(config_path)
    
    @staticmethod   
    def update_config_from_args(config: BaseConfig, args: argparse.Namespace) -> BaseConfig:
        """
        更新配置文件参数从命令行参数。
        参数:
            config (BaseConfig): 配置文件对象，包含默认参数。
            args (argparse.Namespace): 命令行参数对象。
        返回:
            BaseConfig: 更新后的配置文件对象，包含合并后的参数。
        """
        for field in fields(config):
            field_name = field.name
            if hasattr(args, field_name):
                setattr(config, field_name, getattr(args, field_name))
        return config
    
    @staticmethod
    def save_config(config: BaseConfig, save_path: Path) -> None:
        """
        将配置文件参数保存到文件。
        参数:
            config (BaseConfig): 配置文件对象，包含参数。
            save_path (Path): 保存路径。
        """
        config.save_config(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()



