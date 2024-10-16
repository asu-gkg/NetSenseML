import socket
import os
import subprocess

# 定义节点的 IP 地址和对应的 rank
ip_rank_map = {
    "192.168.1.170": 0,
    "192.168.1.154": 1,
    "192.168.1.157": 2,
    "192.168.1.107": 3,
    "192.168.1.109": 4,
    "192.168.1.232": 5,
    "192.168.1.199": 6,
    "192.168.1.248": 7,
}

# 获取当前主机的 IP 地址
def get_current_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        return "127.0.0.1"

# 根据 IP 地址获取当前节点的 rank
def get_rank_from_ip(ip_address):
    rank = ip_rank_map.get(ip_address)
    if rank is None:
        raise ValueError(f"无法为IP地址 {ip_address} 分配 rank，请确保IP地址在映射表中。")
    return rank

# 运行 torchrun 命令
def run_torchrun(rank, master_addr="192.168.1.170", master_port=4003, nproc_per_node=8, nnodes=8):
    log_file = f"./results/benchmark_vgg16_ours_{rank}.log"
    
    # 构造 torchrun 命令
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "app/dist_vgg16.py",
        f"--log_file={log_file}"
    ]
    
    # 打印并运行 torchrun 命令
    print(f"运行 torchrun 命令: {' '.join(torchrun_cmd)}")
    subprocess.run(torchrun_cmd)

def main():
    try:
        # 获取当前主机的 IP 地址
        current_ip = get_current_ip()
        print(f"当前主机的IP地址: {current_ip}")

        # 获取当前主机的 rank
        rank = get_rank_from_ip(current_ip)
        print(f"为主机 {current_ip} 分配的 rank 是: {rank}")

        # 运行 torchrun
        run_torchrun(rank)
        
    except Exception as e:
        print(f"运行 torchrun 时发生错误: {e}")

if __name__ == "__main__":
    main()