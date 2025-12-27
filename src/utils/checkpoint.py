import os

import torch


def load_checkpoint(path, model, optimizer, logger):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint path {path}from epoch {checkpoint['epoch']}")

    return checkpoint["epoch"]


def save_checkpoint(model, optimizer, epoch, step, metrics, experiment_id, is_best=False):
    ckpt_filename = f"{config.model.name}_{config.experiment_id}_epoch_{epoch + 1}.pth"
    ckpt_dir = os.path.join(config.exp_dir, "checkpoints", ckpt_filename)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 步骤2：保存ckpt（同方案1）
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, ckpt_dir)

    # 步骤3：生成meta文件（记录全量信息）
    meta_info = {
        "ckpt_filename": ckpt_filename,
        "experiment_id": experiment_id,
        "model_name": "ResNet50",
        "epoch": epoch,
        "step": step,
        "metrics": metrics,  # 全量指标（acc/loss/precision/recall等）
        "hyperparameters": config["hyperparameters"],  # 超参（从config读取）
        "git_hash": get_git_hash(),  # 代码版本（复用你之前的工具函数）
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_env": config.get("env", "local"),  # 训练环境
        "data_path": config.get("data_path"),  # 数据路径
    }

    # 保存meta文件
    meta_filename = ckpt_filename.replace(".pth", ".meta.json")
    meta_path = os.path.join(save_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)

    print(f"✅ Checkpoint已保存：{ckpt_path}")
    print(f"✅ Meta文件已保存：{meta_path}")

    last_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
        last_path,
    )

    if is_best:
        best_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            best_path,
        )
