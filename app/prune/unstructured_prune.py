import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision

def l1_unstructured_prune_model(model, amount=0.3):
    """
    对模型的所有卷积层和全连接层进行非结构化剪枝。
    Args:
        model: 需要剪枝的模型
        amount: 剪枝比例（例如，0.3 表示剪掉 30% 的参数）
    """
    # 对模型中的每个卷积层和全连接层进行剪枝
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 移除剪枝参数，固定剪枝
            prune.remove(module, 'weight')
    return model