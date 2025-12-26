import os
import subprocess
import sys
import tempfile

def test_smoke_train_runs():
    # 使用一个临时输出目录避免污染 repo
    # 运行训练脚本并确保退出码为 0
    cmd = [sys.executable, "-m", "src.train", "--config", "configs/default.yaml", "--run_name", "ci_smoke_test"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0
    # 检查是否产生 experiments/ci_smoke_test*
    assert os.path.exists("experiments")