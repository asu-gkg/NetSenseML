import torch.distributed as dist
import torch
from ccl.gdc_compression import SimpleDGCCompressor
import time
import logging
from ccl.compensator import SparsifyCompensator
from ccl.topk import TopKCompressor




def default_comm_hook(state, bucket):
    tensor = bucket.buffer()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    fut = torch.futures.Future()
    fut.set_result(tensor / dist.get_world_size())
    return fut

last_rtt = None
smooth_rtt = None
alpha = 0.1 # 平滑系数
compress_ratio = 0.75

def aimd(rtt):
    global last_rtt, compress_ratio, smooth_rtt
    if smooth_rtt is None or last_rtt is None:
        smooth_rtt = rtt
        last_rtt = rtt
        return compress_ratio
    
    smooth_rtt = alpha * rtt + (1 - alpha) * smooth_rtt

    if smooth_rtt < last_rtt:
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


previous_throughput = None
previous_rtt = None
min_rtt = float('inf')
max_bandwidth = 0
last_reset_time = time.perf_counter() 

rtt_window = 5  

def estimate_bandwidth(data_size, rtt):
    return data_size / rtt 

def update_min_rtt_and_bandwidth(rtt, bandwidth, rtt_window):
    global min_rtt, max_bandwidth, last_reset_time
    
    current_time = time.perf_counter()
    
    if current_time - last_reset_time >= rtt_window:
        min_rtt = float('inf')
        max_bandwidth = 0
        last_reset_time = current_time 
    
    # update min_rtt and max_bandwidth
    min_rtt = min(min_rtt, rtt)
    max_bandwidth = max(max_bandwidth, bandwidth)
    
def calculate_bdp(bandwidth, rtt):
    return bandwidth * rtt

def update_compression_ratio(rtt, bandwidth, data_in_flight):
    global compress_ratio, max_bandwidth, min_rtt
    if min_rtt==float('inf') or max_bandwidth==0:
        return compress_ratio
    bdp = calculate_bdp(max_bandwidth, min_rtt)
    
    # based on BDP to change compression ratio
    if data_in_flight > bdp * 0.9:
        compress_ratio = max(0.005, compress_ratio * 0.95)
    else:
        compress_ratio = min(1, compress_ratio + 0.001)
    
    return compress_ratio

def adaptive_bbr_comm_hook(state, bucket):
    global compress_ratio
    print(compress_ratio)
    if compress_ratio == 1:
        return default_comm_hook(state, bucket)

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
        dist.all_gather(gathered_values, values[0: length_])
        dist.all_gather(gathered_indices, indices[0: length_])

        combined_values = torch.cat(gathered_values)
        combined_indices = torch.cat(gathered_indices)
        return combined_values, combined_indices

    start_time = time.perf_counter()
    combined_values, combined_indices = gather()
    end_time = time.perf_counter()
    
    rtt = end_time - start_time
    data_size = combined_values.numel() * combined_values.element_size()
    bandwidth = estimate_bandwidth(data_size, rtt)
    data_in_flight = data_size  # 在途数据量等于传输的数据量

    elapsed_time = end_time % rtt_window
    
    update_min_rtt_and_bandwidth(rtt, bandwidth, elapsed_time)

    compress_ratio = update_compression_ratio(rtt, bandwidth, data_in_flight)

    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut

def common_gather(values, indices):
    gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
    gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_values, values)
    dist.all_gather(gathered_indices, indices)
    combined_values = torch.cat(gathered_values)
    combined_indices = torch.cat(gathered_indices)
    return combined_values, combined_indices

def dgc_comm_hook(state: SparsifyCompensator, bucket: torch.distributed.GradBucket):
    tensor = bucket.buffer()
    param_name = f"param_{bucket.index}_{tensor.size}"
    
    compensated_tensor = state.compensate(tensor, param_name)
    
    compressor = SimpleDGCCompressor(compress_ratio=0.005)
    values, indices, numel = compressor.compress(compensated_tensor)
    combined_values, combined_indices = common_gather(values, indices)
    decompressed_tensor = compressor.decompress((combined_values, combined_indices, numel), tensor.size())
    state.update_residual(tensor, decompressed_tensor, param_name)
    
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor / dist.get_world_size())
    return fut