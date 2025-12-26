import argparse
import json
import os

import torch

from src.data_pipeline.preprocessing import preprocess_data
from src.models.informer import Informer
from src.test import test
from src.trainer import train
from src.training.exp_informer import train_model  # 假设你把训练循环封装成函数

# 不管是本地、CI 还是集群，都用同一套命令：
# # 本地训练ResNet50
# python launch.py --task train --config configs/resnet50.json --env local

# # CI环境测试ViT
# python launch.py --task test --config configs/vit.json --env ci

# # 集群训练ResNet50
# python launch.py --task train --config configs/resnet50.json --env cluster


def load_config(config_path):
    """加载configs/目录下的配置文件"""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    # 1. 定义统一的命令行参数（所有任务共用）
    parser = argparse.ArgumentParser(description="统一实验启动器（launcher）")
    parser.add_argument("--task", required=True, choices=["train", "test"], help="任务类型：训练/测试")
    parser.add_argument("--config", required=True, help="配置文件路径，如configs/resnet50.json")
    parser.add_argument("--env", default="local", choices=["local", "ci", "cluster"], help="运行环境")

    args = parser.parse_args()

    # 2. 加载配置（标准化参数来源）
    config = load_config(args.config)

    # 3. 适配不同运行环境（封装环境相关逻辑）
    if args.env == "cluster":
        # 集群环境：比如设置分布式训练参数、集群存储路径
        config["distributed"] = True
        config["data_path"] = "/cluster/datasets/imagenet"
    elif args.env == "ci":
        # CI环境：比如使用小规模数据集、缩短训练轮数
        config["epochs"] = 1
        config["batch_size"] = 8
    else:
        # 本地环境：使用本地路径
        config["data_path"] = "./datasets/imagenet"

    # 4. 执行对应任务（统一调用逻辑）
    if args.task == "train":
        train(config)
    elif args.task == "test":
        test(config)


if __name__ == "__main__":
    main()


import argparse
import json
import os

from src.tester import test
from src.trainer import train
from utils.experiment_recorder import record_experiment_info


def load_config(config_path):
    """加载configs目录下的配置文件（支持相对/绝对路径）"""
    # 处理相对路径（兼容任意目录下调用launcher）
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    # 校验配置文件存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def adapt_env(config, env):
    """适配不同运行环境（本地/CI/集群）"""
    if env == "cluster":
        # 集群环境：分布式训练、集群存储路径、多GPU配置
        config["data_path"] = "/cluster/datasets/imagenet"  # 集群数据路径
        config["hyperparameters"]["distributed"] = True
        config["hyperparameters"]["gpu_ids"] = [0, 1, 2, 3]  # 集群GPU
    elif env == "ci":
        # CI环境：小规模数据、短训练轮数、快速验证
        config["data_path"] = "./ci_test_data"  # CI测试数据路径
        config["hyperparameters"]["epochs"] = 1  # 仅训练1轮
        config["hyperparameters"]["batch_size"] = 8  # 小批量
        config["hyperparameters"]["gpu_ids"] = []  # CI可能无GPU
    else:  # local
        # 本地环境：本地路径、单GPU
        config["data_path"] = "./datasets/imagenet"  # 本地数据路径
        config["hyperparameters"]["gpu_ids"] = [0]  # 本地GPU

    return config


def main():
    # 1. 定义统一命令行参数（所有任务共用）
    parser = argparse.ArgumentParser(
        description="统一实验启动器（替代ad-hoc脚本，适配CI/集群）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=["train", "test"], help="任务类型：训练/测试")
    parser.add_argument("--config", required=True, help="配置文件路径，如configs/resnet50_train.json")
    parser.add_argument("--env", default="local", choices=["local", "ci", "cluster"], help="运行环境：本地/CI/集群")
    parser.add_argument("--record", action="store_true", default=True, help="是否记录实验信息（git hash + config）")

    # 解析参数
    args = parser.parse_args()

    try:
        # 2. 加载并适配配置
        config = load_config(args.config)
        config = adapt_env(config, args.env)

        # 3. 记录实验信息（git hash + config）
        if args.record:
            record_experiment_info(config, args.env)

        # 4. 执行对应任务（统一调用逻辑）
        if args.task == "train":
            train(config)
        elif args.task == "test":
            test(config)

    except Exception as e:
        print(f"❌ 任务执行失败：{str(e)}")
        raise  # 抛出异常，便于CI/集群捕获失败


if __name__ == "__main__":
    main()
