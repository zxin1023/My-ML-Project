import pandas as pd
import wandb

api = wandb.Api()
runs = api.runs("your-project-name")


data = []
for run in runs:
    data.append(
        {
            "exp_name": run.name,
            "train_loss": run.summary["train_loss"],
            "val_acc": run.summary["val_acc"],
            "epoch": run.summary["epoch"],
        }
    )

df = pd.DataFrame(data)
df.to_csv("experiments/result.csv", index=False)
