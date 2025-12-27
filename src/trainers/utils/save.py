import json
import os


def save_run_meta(output_dir, meta):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
