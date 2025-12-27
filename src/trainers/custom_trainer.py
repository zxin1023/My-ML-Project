#!/usr/bin/env python3
"""
最小化训练脚本示例（可以扩展：数据加载、模型、训练循环、日志）。
在运行时会把 git hash & config 保存到 output_dir/run_meta.json。
"""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

from src.utils.config import get_config
from src.utils.git import get_git_hash
from src.utils.logger import load_config
from src.utils.seed import set_seed
from utils.save import save_run_meta

# 假设项目采用 src 包结构
# from src import data, models, utils


def train(config):
    """训练核心逻辑（替换原有ad-hoc训练脚本）"""
    print("\n========== 开始训练 ==========")
    # 从config读取超参（标准化参数来源）
    experiment_id = config["experiment_id"]
    model = config["model"]
    lr = config["hyperparameters"]["lr"]
    batch_size = config["hyperparameters"]["batch_size"]
    epochs = config["hyperparameters"]["epochs"]
    data_path = config["data_path"]  # 环境适配后的路径

    # 模拟训练逻辑（替换为你的真实训练代码）
    print(f"实验ID：{experiment_id}")
    print(f"模型：{model}")
    print(f"超参：lr={lr}, batch_size={batch_size}, epochs={epochs}")
    print(f"数据路径：{data_path}")
    print("训练中...（加载数据→模型初始化→前向/反向传播→保存模型）")
    print("========== 训练完成 ==========\n")

    # 返回训练结果（可选，用于后续记录）
    return {"status": "success", "best_acc": 0.92}


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.py", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    config = BaseConfig(args.config)

    set_seed(config.get("seed", 42))

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{config.get('run',{}).get('name','run')}_{timestamp}"
    output_dir = os.path.join(config.get("run", {}).get("output_dir", "experiments"), run_name)
    # 记录环境信息
    run_meta = {
        "git_hash": get_git_hash(),
        "config_file": args.config,
        "config": config,
        "timestamp": timestamp,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
    }
    save_run_meta(output_dir, run_meta)

    # 设置随机种子
    seed = config.get("seed", 42)
    set_seed(seed)

    # TODO: 初始化数据集、模型、优化器
    print("Starting run:", run_name)
    print("Saved run meta to:", output_dir)
    # 这里做一个简单的 smoke train loop（示例）
    epochs = config.get("training", {}).get("epochs", 1)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} -- dummy train step")
        # 模拟 checkpoint 保存
        if epoch % config.get("checkpoint", {}).get("save_every_n_epochs", 1) == 0:
            ckpt_path = os.path.join(output_dir, f"ckpt_epoch_{epoch}.pt")
            with open(ckpt_path, "w") as f:
                f.write("dummy checkpoint")
            print("Saved checkpoint:", ckpt_path)
    print("Run finished")


if __name__ == "__main__":
    main()
