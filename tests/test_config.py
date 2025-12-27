import os

import pytest
import yaml

from src.utils.config import load_config


def test_load_config(tmp_path):
    # 1. Setup: Create a dummy config file
    config_data = {"train": {"epochs": 10, "save_dir": str(tmp_path / "checkpoints")}}
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # 2. Action: Call the function
    config = load_config(str(config_file))

    # 3. Assertion: Verify results
    assert config["train"]["epochs"] == 10
    assert os.path.exists(config["train"]["save_dir"])
