from typing import Any, Tuple

import torch


from base_dataset import BaseDataset


class CustomDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data = self.load_data()
        
    def load_data(self):
        """
        加载数据集的方法，所有子类都必须实现该方法。
        """
        raise NotImplementedError("子类必须实现load_data方法")
        
    def preprocess(self, data: Any) -> torch.Tensor:
        """
        预处理数据集的方法，所有子类都必须实现该方法。
        """
        raise NotImplementedError("子类必须实现preprocess方法")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回数据集的第idx个样本。
        """
        return self.data[idx]
