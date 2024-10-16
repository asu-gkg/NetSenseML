import os
import time
import subprocess

def run_iperf_periodic(client_ip, max_p, interval, duration, cycles):
    """
    运行 iperf3 客户端，周期性增加 P 并循环
    :param client_ip: iperf3 服务器的 IP 地址
    :param port: iperf3 服务器端口
    :param max_p: 最大并发流数量
    :param interval: 每次增加 P 后的等待时间（秒）
    :param duration: 每次运行 iperf3 的持续时间（秒）
    :param cycles: 循环次数
    """
    for cycle in range(cycles):
        print(f"\nCycle {cycle + 1}/{cycles} Starting...\n")
        for p in range(1, max_p + 1):
            # 构造 iperf3 命令
            cmd = f"iperf3 -c {client_ip} -t {duration} -P {p}"
            
            print(f"Running iperf3 with P={p} in Cycle {cycle + 1}...")
            # 运行 iperf3 命令
            process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 打印输出结果
            print(process.stdout.decode())
            print(process.stderr.decode())
            
            # 等待指定时间再增加并发流数量
            time.sleep(interval)

if __name__ == "__main__":
    # 设定 iperf3 服务器地址和参数
    client_ip = "192.168.1.199"  # 服务器 IP
    max_p = 50                    # 最大并发流数量
    interval = 10                 # 每次增加 P 的时间间隔（秒）
    duration = 60                 # 每次 iperf3 持续时间（秒）
    cycles = 3000                 # 循环次数

    run_iperf_periodic(client_ip, max_p, interval, duration, cycles)