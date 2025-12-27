import torch


def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / targets.size(0)
