import torch

class SparsifyCompensator:
    def __init__(self, model):
        # 初始化动量和残差字典
        self.momentums = {}
        self.residuals = {}
        self.momentum_factor = 0.9  # 动量因子，通常设为 0.9 或 0.99
        
    def compensate(self, grad, name):
        """通过动量和残差对当前梯度进行补偿"""
        if self.residuals.get(name) is None or self.momentums.get(name) is None:
            self.momentums[name] = torch.zeros_like(grad)
            self.residuals[name] = torch.zeros_like(grad)
        # 加入残差
        grad_with_residual = grad + self.residuals[name]
        
        # 使用动量补偿梯度
        self.momentums[name].mul_(self.momentum_factor).add_(grad_with_residual)
        return self.momentums[name]
    
    def update_residual(self, grad, decompressed_tensor, name):
        """更新残差，保存未传输的梯度部分"""
        self.residuals[name] = grad - decompressed_tensor

