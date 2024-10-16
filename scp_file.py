import subprocess

# 定义节点的连接信息
nodes = {
    "192.168.1.154": {"user": "d", "password": "d"},
    "192.168.1.157": {"user": "ddd", "password": "ddd"},
    "192.168.1.107": {"user": "dddd", "password": "dddd"},
    "192.168.1.109": {"user": "ddddd", "password": "ddddd"},
    "192.168.1.232": {"user": "dddddd", "password": "dddddd"},
    "192.168.1.199": {"user": "ddddddd", "password": "ddddddd"},
    "192.168.1.249": {"user": "dddddddd", "password": "dddddddd"}
}

# 本地需要同步的目录
LOCAL_DIR = "/home/dd/NetSenseML/"

# 函数：通过 rsync 传输文件并显示速度
def transfer_files_rsync(ip, user, local_path, remote_path):
    try:
        # 使用 rsync 进行文件传输
        rsync_command = [
            "rsync", "-avz", "--progress", local_path,
            f"{user}@{ip}:{remote_path}"
        ]
        subprocess.run(rsync_command)
        print(f"成功同步到 {ip}")
    except Exception as e:
        print(f"同步到 {ip} 时出错: {e}")

# 同步文件到每个节点
for ip, info in nodes.items():
    print(f"正在同步文件到 {ip} ({info['user']})...")
    
    # 远程目录
    remote_dir = f"/home/{info['user']}/NetSenseML/"
    
    # 使用 rsync 传输文件
    transfer_files_rsync(ip, info['user'], LOCAL_DIR, remote_dir)