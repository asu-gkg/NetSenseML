import torch
import torch.distributed as dist
from ccl import Memory

class DgcMemory(Memory):
    def __init__(self, momentum, gradient_clipping):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        # 如果需要进行梯度裁剪
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            
            # 替换 allreduce_ 为 torch.distributed.all_reduce
            dist.all_reduce(tensor_squ_sum, op=dist.ReduceOp.SUM)
            # 计算平均值
            clipping_val = torch.sqrt(tensor_squ_sum / dist.get_world_size())
            
            # 裁剪梯度
            tensor = tensor.clamp(-clipping_val, clipping_val)
        
        # 计算动量残差
        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor
        
        # 更新动量和梯度
        if name in self.gradients:
            self.gradients[name] += self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        # 更新残差和梯度中的未压缩部分
        temp = self.residuals[name] * not_mask
        self.residuals[name] = temp
        temp = self.gradients[name] * not_mask
        self.gradients[name] = temp