import torch
import torch.distributed as dist
from ccl import Memory
from ccl.topk import TopKCompressor

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
        
        
class DgcCommHook:
    def __init__(self, momentum=0.9, gradient_clipping=True, compress_ratio=0.01):
        self.dgc_memory = DgcMemory(momentum, gradient_clipping)
        self.compressor = TopKCompressor(compress_ratio)

    def dgc_comm_hook(self, state, bucket: dist.GradBucket):
        # 获取当前梯度的张量
        tensor = bucket.buffer()
        param_name = f"param_{bucket.index}_{tensor.size()}"

        # 使用 DGC 内存进行补偿
        compensated_tensor = self.dgc_memory.compensate(tensor, param_name)

        # 压缩补偿后的梯度
        compressed_tensor, ctx = self.compressor.compress(compensated_tensor, param_name)

        # 执行 all_reduce 操作来同步梯度
        gathered_values = [torch.zeros_like(compressed_tensor[0]) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(compressed_tensor[1]) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_values, compressed_tensor[0])  # Gather values
        dist.all_gather(gathered_indices, compressed_tensor[1])  # Gather indices

        # 聚合并解压梯度
        all_values = torch.cat(gathered_values)
        all_indices = torch.cat(gathered_indices)

        decompressed_tensor = self.compressor.decompress((all_values, all_indices), ctx)

        # 使用 DGC 更新残差
        self.dgc_memory.update(tensor, param_name, self.compressor, decompressed_tensor, ctx)

        # 返回解压后的张量并进行归一化
        fut = torch.futures.Future()
        fut.set_result(decompressed_tensor / dist.get_world_size())
        return fut
