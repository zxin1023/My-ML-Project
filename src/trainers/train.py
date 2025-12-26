#!/usr/bin/env python3
"""
最小训练脚本示例，包含：
- 读取 YAML config
- 记录 run_meta（git hash、config、env）
- 简单的训练循环（示例数据）
- 可选 W&B 集成
- 保存 checkpoint 到 output_dir
用法：
  python -m src.train --config configs/default.yaml --run_name myrun
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

try:
    import wandb
except Exception:
    wandb = None


def get_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_run_meta(output_dir, meta):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def dummy_dataset(n_samples=256, input_dim=32, n_classes=10):
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int64)
    return X, y


def simple_train_loop(config, output_dir):
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    # 如果 data_dir 存在且包含 npz 文件，可尝试加载；否则使用 dummy 数据
    data_dir = dataset_cfg.get("data_dir", "data/processed")
    if os.path.exists(os.path.join(data_dir, "demo_data.npz")):
        data = np.load(os.path.join(data_dir, "demo_data.npz"))
        X, y = data["X"], data["y"]
        print("Loaded data from", os.path.join(data_dir, "demo_data.npz"))
    else:
        print("Using dummy dataset")
        X, y = dummy_dataset(
            n_samples=512,
            input_dim=model_cfg.get("input_dim", 32),
            n_classes=model_cfg.get("num_classes", 10),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 简单模型：线性层
    model = torch.nn.Sequential(
        torch.nn.Linear(model_cfg.get("input_dim", 32), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, model_cfg.get("num_classes", 10)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("lr", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = training_cfg.get("epochs", 1)
    batch_size = training_cfg.get("batch_size", 32)
    n = X.shape[0]
    metrics = {"train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.tensor(X[idx], device=device)
            yb = torch.tensor(y[idx], device=device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss = epoch_loss / n
        metrics["train_loss"].append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} - train_loss: {epoch_loss:.4f}")

        # W&B logging
        if config.get("logging", {}).get("use_wandb", False) and wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": epoch_loss})

        # 保存 checkpoint
        if epoch % config.get("checkpoint", {}).get("save_every_n_epochs", 1) == 0:
            ckpt_path = os.path.join(output_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print("Saved checkpoint:", ckpt_path)

    # 写 metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
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

    # 初始化 W&B（如果配置）
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    if use_wandb:
        if wandb is None:
            print("wandb 未安装，跳过 wandb logging")
        else:
            wandb_cfg = config.get("logging", {})
            wandb_run = wandb.init(
                project=wandb_cfg.get("wandb_project"),
                entity=wandb_cfg.get("wandb_entity") or None,
                name=run_name,
                config=config,
                reinit=True,
            )
            run_meta["wandb_run_id"] = wandb_run.id

    print("Starting run:", run_name)
    print("Saved run meta to:", output_dir)

    metrics = simple_train_loop(config, output_dir)

    if use_wandb and wandb is not None:
        wandb.log({"final_train_loss": metrics["train_loss"][-1]})
        wandb.finish()

    print("Run finished")


if __name__ == "__main__":
    main()