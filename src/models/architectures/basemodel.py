
import torch
from torch import nn
from typing import Dict, Any


class BaseModel(nn.Module):
    def __init__(self, config: 'BaseConfig'):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        raise NotImplementedError
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'parameter_count': sum(p.numel() for p in self.parameters()),
                'trainable_count': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def load_weights(self, weights_path: str):
        self.load_state_dict(torch.load(weights_path))