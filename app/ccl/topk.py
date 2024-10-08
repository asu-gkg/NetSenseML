import torch
from ccl import Compressor
import torch.distributed as dist


def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed

class TopKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)
    
    

def topk_comm_hook(state, bucket):
    tensor = bucket.buffer()
    
    compressor = TopKCompressor(compress_ratio=0.1)
    compressed_tensors, ctx = compressor.compress(tensor, f"param_{bucket.index}")
    
    gathered_tensors = [torch.zeros_like(compressed_tensors[0]) for _ in range(dist.get_world_size())]
    gathered_indices = [torch.zeros_like(compressed_tensors[1]) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_tensors, compressed_tensors[0])  # gather values
    dist.all_gather(gathered_indices, compressed_tensors[1])  # gather indices

    all_values = torch.cat(gathered_tensors)
    all_indices = torch.cat(gathered_indices)

    decompressed_tensor = compressor.decompress((all_values, all_indices), ctx)

    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut