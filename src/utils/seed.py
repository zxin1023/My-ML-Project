import random
import numpy as np
import torch


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    # 可选：torch.backends.cudnn.deterministic = True
    # 可选：torch.backends.cudnn.benchmark = False