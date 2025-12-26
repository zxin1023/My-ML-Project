import torch
from typing import Any, Tuple

from abc import ABC
from torch.utils.data import Dataset
from abc import abstractmethod


class BaseDataset(Dataset, ABC):
    """
    基础数据集类，所有数据集都应继承自该类。
    """
    def __init__(self, config: 'BaseConfig'):
        self.config = config

    @abstractmethod
    def load_data(self):
        """
        加载数据集的方法，所有子类都必须实现该方法。
        """
        pass
        raise NotImplementedError("子类必须实现load_data方法")
    
    @abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """
        预处理数据集的方法，所有子类都必须实现该方法。
        """
        raise NotImplementedError("子类必须实现preprocess方法")
    
    @abstractmethod
    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回数据集的第idx个样本。
        """
        return self.data[idx]