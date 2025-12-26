#!/usr/bin/env python3
"""
DVC prepare stage 的示例脚本：
- 如果你有真实数据：在这里实现数据下载 / 解压 / 处理，并把结果写到 data/processed
- 如果没有真实数据：本脚本会生成一个小型演示数据集 data/processed/demo_data.npz
用法：
  python scripts/prepare_data.py
"""
import os
import numpy as np


def prepare_demo_data(out_dir="data/processed", n_samples=512, input_dim=32, n_classes=10):
    os.makedirs(out_dir, exist_ok=True)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int64)
    np.savez_compressed(os.path.join(out_dir, "demo_data.npz"), X=X, y=y)
    print("Wrote demo data to", os.path.join(out_dir, "demo_data.npz"))


def main():
    # TODO: Replace this with real data download/processing if needed
    prepare_demo_data()


if __name__ == "__main__":
    main()