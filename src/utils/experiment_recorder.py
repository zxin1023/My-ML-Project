import json
import os
import subprocess
from datetime import datetime


def get_git_hash():
    """获取当前代码的git commit哈希（唯一标识代码版本）"""
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, cwd=os.getcwd())
            .decode("utf-8")
            .strip()
        )
        return git_hash
    except subprocess.CalledProcessError:
        return "no_git_repo"  # 非git仓库时返回默认值
    except FileNotFoundError:
        return "git_not_installed"  # 未安装git时返回默认值


def record_experiment_info(config, env, save_path="./experiment_records"):
    """记录实验信息（git hash + config + 环境 + 时间）"""
    # 创建记录目录
    os.makedirs(save_path, exist_ok=True)

    # 组装记录信息
    record = {
        "git_hash": get_git_hash(),
        "config": config,
        "env": env,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": config.get("experiment_id", "unknown"),
    }

    # 保存记录（按实验ID+时间命名，避免覆盖）
    record_filename = f"{record['experiment_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    record_path = os.path.join(save_path, record_filename)

    with open(record_path, "w") as f:
        json.dump(record, f, indent=4, ensure_ascii=False)

    print(f"✅ 实验记录已保存至：{record_path}")
    return record
