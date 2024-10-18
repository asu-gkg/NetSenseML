from fabric import Connection
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 节点信息，包括远程主机的IP地址、用户名、密码和用户的NetSenseML目录
nodes = {
    "192.168.1.154": {"user": "d", "password": "d", "remote_directory": "/home/d/NetSenseML"},
    "192.168.1.169": {"user": "dd", "password": "dd", "remote_directory": "/home/dd/NetSenseML"},
    "192.168.1.157": {"user": "ddd", "password": "ddd", "remote_directory": "/home/ddd/NetSenseML"},
    "192.168.1.108": {"user": "dddd", "password": "dddd", "remote_directory": "/home/dddd/NetSenseML"},
    "192.168.1.107": {"user": "ddddd", "password": "ddddd", "remote_directory": "/home/ddddd/NetSenseML"},
    "192.168.1.232": {"user": "dddddd", "password": "dddddd", "remote_directory": "/home/dddddd/NetSenseML"},
    "192.168.1.199": {"user": "ddddddd", "password": "ddddddd", "remote_directory": "/home/ddddddd/NetSenseML"},
    "192.168.1.248": {"user": "dddddddd", "password": "dddddddd", "remote_directory": "/home/dddddddd/NetSenseML"}
}

# 当前主机的 rank 和 world_size
current_host_rank = 1  # 当前主机的 rank
world_size = len(nodes)

# 动态命令模板，用户可以在运行时修改它
command_template = "cd {remote_directory} && make run-dist-resnet18 world_size={world_size} rank={rank}"

# 执行 SSH 命令的函数，使用 fabric 来代替 paramiko
def fabric_execute_command(hostname, username, password, remote_directory, command):
    
    try:
        # 创建 Fabric 连接
        conn = Connection(host=hostname, user=username, connect_kwargs={"password": password})
        
        # 连接远程主机并执行命令
        print(f"正在连接到 {hostname}...")
        hide = True
        if username=='dd':
            hide = False
        result = conn.run(command, hide=hide, pty=True)
        
        # 实时打印命令输出
        print(f"[{hostname}] {result.stdout}")
        if result.stderr:
            print(f"[{hostname} ERROR] {result.stderr}")
        
    except Exception as e:
        print(f"在主机 {hostname} 执行命令时出错: {e}")
    finally:
        # 关闭连接
        conn.close()

# 并行执行远程和本地命令
def run_commands_in_parallel():
    with ThreadPoolExecutor() as executor:
        futures = []
        # 为每个远程主机添加任务
        rank = 0 
        for hostname, credentials in nodes.items():
            username = credentials['user']
            password = credentials['password']
            remote_directory = credentials['remote_directory']
            
            # 根据模板生成远程主机要执行的命令
            command = command_template.format(
                remote_directory=remote_directory, world_size=world_size, rank=rank)
            
            # 提交远程任务
            futures.append(executor.submit(fabric_execute_command, hostname, username, password, remote_directory, command))
            
            rank += 1  # 每次循环后增加 rank 值

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"任务执行失败: {e}")

# 并行执行本地和远程命令
run_commands_in_parallel()

print("分布式计算命令在所有主机上执行完成。")
