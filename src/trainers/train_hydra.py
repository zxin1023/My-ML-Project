#!/usr/bin/env python3
"""
Hydra training entry (config_path="conf").
用法示例：
  python train_hydra.py
  python train_hydra.py model.input_dim=64 training.epochs=5
  python train_hydra.py -m training.lr=0.01,0.001
"""
import os
import subprocess
import sys
from datetime import datetime

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except Exception:
    wandb = None


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dummy_dataset(n_samples=512, input_dim=32, n_classes=10):
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int64)
    return X, y


def simple_train_loop(cfg: DictConfig, output_dir: str):
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    training_cfg = cfg.training

    data_dir = dataset_cfg.get("data_dir", "data/processed")
    demo_file = os.path.join(data_dir, "demo_data.npz")
    if os.path.exists(demo_file):
        data = np.load(demo_file)
        X, y = data["X"], data["y"]
        print("Loaded data from", demo_file)
    else:
        print("Using dummy dataset")
        X, y = dummy_dataset(
            n_samples=512,
            input_dim=model_cfg.get("input_dim", 32),
            n_classes=model_cfg.get("num_classes", 10),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if cfg.logging.get("use_wandb", False) and wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": epoch_loss})

        # checkpoint
        if epoch % cfg.checkpoint.get("save_every_n_epochs", 1) == 0:
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

    # write metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    return metrics


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    repo_root = get_original_cwd()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_name = cfg.run.get("name", "hydra_run")
    run_name = f"{short_name}_{timestamp}"

    output_dir = os.path.join(repo_root, cfg.run.get("output_dir", "experiments"), run_name)
    os.makedirs(output_dir, exist_ok=True)

    OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))

    run_meta = {
        "git_hash": get_git_hash(),
        "config_file": "conf/config.yaml",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "timestamp": timestamp,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
    }
    import json

    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    use_wandb = cfg.logging.get("use_wandb", False)
    if use_wandb:
        if wandb is None:
            print("wandb not installed, skipping wandb logging")
        else:
            wandb_cfg = cfg.logging
            wandb_run = wandb.init(
                project=wandb_cfg.get("wandb_project"),
                entity=wandb_cfg.get("wandb_entity") or None,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            run_meta["wandb_run_id"] = wandb_run.id
            with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
                json.dump(run_meta, f, indent=2)

    print("Starting run:", run_name)
    print("Output dir:", output_dir)

    metrics = simple_train_loop(cfg, output_dir)

    if use_wandb and wandb is not None:
        wandb.log({"final_train_loss": metrics["train_loss"][-1]})
        wandb.finish()

    print("Run finished")


if __name__ == "__main__":
    main()
