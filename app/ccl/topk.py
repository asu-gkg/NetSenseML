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



class TopKCompressor:
    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio

    def _sparsify(self, tensor):
        # Flatten the tensor
        tensor = tensor.view(-1)
        numel = tensor.numel()
        num_selects = max(1, int(numel * self.compress_ratio))

        # Get the importance (absolute values)
        importance = tensor.abs()

        # Find the top k% elements directly using topk
        threshold, indices = torch.topk(importance, num_selects, sorted=False)

        # Extract the values at these indices
        values = tensor[indices]
        return values, indices, numel

    def compress(self, tensor):
        # Perform sparsification
        values, indices, numel = self._sparsify(tensor)
        
        # Return the compressed representation and the original tensor size
        return (values, indices, numel)

    def decompress(self, compressed_tensor, original_size):
        values, indices, numel = compressed_tensor
        # Create an empty tensor of the original size
        decompressed_tensor = torch.zeros(original_size, device=values.device)
        # Put the values back to their original positions
        decompressed_tensor.index_put_([indices], values)
        # Reshape back to the original shape
        return decompressed_tensor.view(original_size)
    

def common_gather(values, indices):
    gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
    gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_values, values)
    dist.all_gather(gathered_indices, indices)
    combined_values = torch.cat(gathered_values)
    combined_indices = torch.cat(gathered_indices)
    return combined_values, combined_indices


def topk_comm_hook(state, bucket):
    tensor = bucket.buffer()

    # Create a compressor and compress the tensor
    compressor = TopKCompressor(compress_ratio=0.005)
    values, indices, numel = compressor.compress(tensor)

    combined_values, combined_indices = common_gather(values, indices)

    # Decompress the tensor with combined values and indices
    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())

    # Return the decompressed tensor divided by world size
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut