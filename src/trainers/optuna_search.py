#!/usr/bin/env python3
"""
A simple Optuna HPO script that launches train_hydra.py as subprocesses with overridden hyperparameters.
This avoids embedding Hydra programmatically in Optuna and keeps each trial isolated.
"""
import argparse
import json
import os
import subprocess

import optuna


def run_trial(trial):
    # Sample hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    run_name = f"optuna_trial_{trial.number}"
    cmd = [
        "python",
        "train_hydra.py",
        f"training.lr={lr}",
        f"training.batch_size={batch_size}",
        f"run.name={run_name}",
        "logging.use_wandb=false",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    print(proc.stderr)

    metrics_path = os.path.join("experiments", run_name, "metrics.json")
    if not os.path.exists(metrics_path):
        # If run failed, return large loss
        return float("inf")
    with open(metrics_path) as f:
        metrics = json.load(f)
    final_loss = metrics.get("train_loss", [float("inf")])[-1]
    return final_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(run_trial, n_trials=args.n_trials)

    print("Best trial:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()