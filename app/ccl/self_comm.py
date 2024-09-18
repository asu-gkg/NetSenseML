import torch.distributed as dist
import torch
from ccl.gdc_compression import SimpleDGCCompressor
import time
import logging

def sparsify_comm_hook(state, bucket):
    tensor = bucket.buffer()

    # Create a compressor and compress the tensor
    compressor = SimpleDGCCompressor(compress_ratio=0.005)
    values, indices, numel = compressor.compress(tensor)


    def gather():
        # Gather values and indices from all processes
        gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]

        # All gather values and indices
        dist.all_gather(gathered_values, values)
        dist.all_gather(gathered_indices, indices)

        # Combine gathered results
        combined_values = torch.cat(gathered_values)
        combined_indices = torch.cat(gathered_indices)
        return combined_values, combined_indices

    combined_values, combined_indices = gather()

    # Decompress the tensor with combined values and indices
    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())

    # Return the decompressed tensor divided by world size
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut


def default_comm_hook(state, bucket):
    tensor = bucket.buffer()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    fut = torch.futures.Future()
    fut.set_result(tensor / dist.get_world_size())
    return fut

last_rtt = None
smooth_rtt = None
alpha = 0.1 # 平滑系数
compress_ratio = 0.01

def aimd(rtt):
    global last_rtt, compress_ratio, smooth_rtt
    if smooth_rtt is None or last_rtt is None:
        smooth_rtt = rtt
        last_rtt = rtt
        return compress_ratio
    
    smooth_rtt = alpha * rtt + (1 - alpha) * smooth_rtt

    # 判断是否增大或减小压缩率
    if smooth_rtt < last_rtt:
        # 自适应增量调整
        compress_ratio = min(compress_ratio + 0.001, 1)
    else:
        compress_ratio = max(compress_ratio * 0.95, 0.005)

    last_rtt = smooth_rtt
    return compress_ratio

def adaptive_sparsify_comm_hook(state, bucket):
    global compress_ratio
    if compress_ratio == 1:
        return default_comm_hook(state, bucket)

    # logging.info(f"Compressed ratio: {compress_ratio}, RTT: {last_rtt}")

    tensor = bucket.buffer()
    compressor = SimpleDGCCompressor(compress_ratio)
    values, indices, numel = compressor.compress(tensor)

    def gather():
        length_ = torch.tensor([len(values)], device=tensor.device)
        dist.all_reduce(length_, op=dist.ReduceOp.MIN)

        gathered_values = []
        gathered_indices = []
        for i in range(dist.get_world_size()):
            gathered_values.append(torch.zeros(length_, dtype=values.dtype, device=tensor.device))
            gathered_indices.append(torch.zeros(length_, dtype=indices.dtype, device=tensor.device))
        # All gather values and indices
        dist.all_gather(gathered_values, values[0: length_])
        dist.all_gather(gathered_indices, indices[0: length_])

        # Combine gathered results
        combined_values = torch.cat(gathered_values)
        combined_indices = torch.cat(gathered_indices)
        return combined_values, combined_indices
    

    start_time = time.perf_counter()

    combined_values, combined_indices = gather()

    end_time = time.perf_counter()
    rtt = end_time - start_time
    compress_ratio = aimd(rtt)

    # Decompress the tensor with combined values and indices
    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())
    
    # Return the decompressed tensor divided by world size
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut

