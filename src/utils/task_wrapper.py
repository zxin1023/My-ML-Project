import os
from datetime import datetime

from utils.config_loader import load_config
from utils.experiment_recorder import get_git_hash  # 复用git hash获取函数


def task_wrapper(task_func):
    """
    通用任务包装器：自动加载配置、记录git hash与config
    :param task_func: 被包装的核心任务函数（如train、test）
    :return: 包装后的任务函数
    """

    def wrapper(config_path, *args, **kwargs):
        # 1. 加载配置文件（从configs/目录）
        config = load_config(config_path)
        print(f"✅ 配置文件加载成功：{config_path}")

        # 2. 获取当前代码的git hash（记录代码版本）
        git_hash = get_git_hash()
        config["git_hash"] = git_hash  # 将git hash写入config，便于后续记录
        print(f"✅ 当前代码git版本：{git_hash}")

        # 3. 自动记录config与git hash（生成记录文件，便于追溯）
        record_dir = "./experiment_records"
        os.makedirs(record_dir, exist_ok=True)
        # 记录文件名：实验ID+时间戳
        record_filename = f"{config['experiment_id']}_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        record_path = os.path.join(record_dir, record_filename)
        # 写入记录（含config、git hash、启动时间）
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": config["experiment_id"],
                    "git_hash": git_hash,
                    "config": config,
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "task_type": task_func.__name__,  # 任务类型（train/test）
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        print(f"✅ 实验记录已保存：{record_path}")

        # 4. 执行核心任务（如train/test），传入config
        print(f"\n===== 开始执行任务：{task_func.__name__} =====")
        result = task_func(config, *args, **kwargs)

        # 5. 任务结束后补充记录（可选）
        print(f"===== 任务执行完成：{task_func.__name__} =====")
        with open(record_path, "r+", encoding="utf-8") as f:
            record = json.load(f)
            record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record["task_result"] = result
            f.seek(0)
            json.dump(record, f, indent=4, ensure_ascii=False)

        return result

    return wrapper


import torch

from utils.checkpoint_saver import save_checkpoint  # 之前的规范命名保存函数
from utils.task_wrapper import task_wrapper  # 导入wrapper


# 用wrapper包装核心训练函数
@task_wrapper
def train(config):
    """
    核心训练函数（被wrapper包装，自动获取config与git hash）
    :param config: 从configs/加载的配置字典（由wrapper传入）
    """
    # 从config中读取超参（无需硬编码）
    experiment_id = config["experiment_id"]
    model_name = config["model"]
    lr = config["hyperparameters"]["lr"]
    batch_size = config["hyperparameters"]["batch_size"]
    epochs = config["hyperparameters"]["epochs"]
    ckpt_save_dir = config["ckpt_save_dir"]
    data_path = config["data_path"]

    # 打印训练信息（验证config加载正常）
    print(f"实验ID：{experiment_id}")
    print(f"模型：{model_name}")
    print(f"超参：lr={lr}, batch_size={batch_size}, epochs={epochs}")
    print(f"数据路径：{data_path}")

    # ----------------------
    # 核心训练逻辑（无需修改）
    # ----------------------
    # 1. 模拟模型初始化、数据加载等
    model = torch.nn.Linear(1000, 10)  # 示例模型（替换为真实ResNet50）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["hyperparameters"]["weight_decay"])

    # 2. 训练循环
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} 训练中...")
        # 模拟训练过程（替换为真实训练代码）
        train_loss = 0.15 + epoch * 0.001
        val_acc = 0.85 + epoch * 0.0014
        best_acc = max(best_acc, val_acc)

        print(f"Epoch {epoch+1} 结果：train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        # 3. 按规范命名保存checkpoint（关联之前的命名规范）
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=epoch * 1000,  # 模拟step计算
            metrics={"train_loss": train_loss, "val_acc": val_acc},
            experiment_id=experiment_id,
            save_dir=ckpt_save_dir,
        )

    # 返回训练结果（由wrapper记录到实验文件）
    return {"status": "success", "experiment_id": experiment_id, "best_acc": best_acc, "git_hash": config["git_hash"]}


# 运行入口（只需传入config文件路径）
if __name__ == "__main__":
    # 传入configs目录下的训练配置文件
    train(config_path="configs/resnet50/train.json")
