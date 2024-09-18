import re
import datetime
import matplotlib.pyplot as plt


def parse_log_file_with_time(log_file):
    epochs = []
    losses = []
    accuracies = []
    compressed_ratios = []
    rtts = []
    times = []
    compression_times = []

    # 定义正则表达式来匹配日志中的数据
    epoch_pattern = re.compile(r'Epoch (\d+)/\d+, loss: ([\d.]+), accuracy: ([\d.]+)')
    compression_pattern = re.compile(r'Compressed ratio: ([\d.]+), RTT: ([\d.]+)')

    with open(log_file, 'r') as file:
        start_time = None
        for line in file:
            # 提取时间戳
            time_str = line.split(' - ')[0]
            current_time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')

            # 如果是第一行，记录开始时间
            if start_time is None:
                start_time = current_time

            # 计算相对时间（秒）
            elapsed_time = (current_time - start_time).total_seconds()
            
            # 匹配 epoch, loss, accuracy
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                accuracy = float(epoch_match.group(3))

                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(accuracy)
                times.append(elapsed_time)

            # 匹配 compressed ratio 和 RTT
            compression_match = compression_pattern.search(line)
            if compression_match:
                compressed_ratio = float(compression_match.group(1))
                rtt = float(compression_match.group(2))

                compressed_ratios.append(compressed_ratio)
                rtts.append(rtt)
                compression_times.append(elapsed_time)

    return epochs, losses, accuracies, compressed_ratios, rtts, times, compression_times

# 示例用法
log_file = 'benchmark_vgg16_0.log'
epochs, losses, accuracies, compressed_ratios, rtts, times, compression_times = parse_log_file_with_time(log_file)

# 绘制时间与准确率的关系图并保存
plt.figure(figsize=(10, 5))
plt.plot(times, accuracies, label='Accuracy', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')
plt.legend()
plt.savefig('./results/vgg16_accuracy_over_time.png')  # 保存图像
plt.close()  # 关闭图像以释放内存


import matplotlib.pyplot as plt

def plot_rtt_and_ratio_over_time(times, rtts, compressed_ratios):
    # 绘制 RTT 和压缩比在同一图表上，使用时间作为 x 轴
    plt.figure(figsize=(10, 5))

    # 绘制压缩比
    plt.plot(compression_times, compressed_ratios, label='Compressed Ratio', color='blue')
    
    # 绘制 RTT
    plt.plot(compression_times, rtts, label='RTT', color='red')

    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('RTT and Compressed Ratio Over Time')
    plt.legend()
    plt.savefig('rtt_and_compressed_ratio_over_time.png')  # 保存图像
    plt.close()  # 关闭图像以释放内存

def plot_rtt_vs_ratio(rtts, compressed_ratios):
    # 绘制 RTT 和压缩比的关系图，使用 RTT 作为 x 轴
    plt.figure(figsize=(10, 5))

    # 绘制 RTT 和压缩比的关系
    plt.scatter(rtts, compressed_ratios, label='Compressed Ratio vs RTT', color='purple')

    plt.xlabel('RTT (s)')
    plt.ylabel('Compressed Ratio')
    plt.title('Compressed Ratio vs RTT')
    plt.legend()
    plt.savefig('compressed_ratio_vs_rtt.png')  # 保存图像
    plt.close()  # 关闭图像以释放内存

# 使用解析得到的数据来绘图
# times, rtts 和 compressed_ratios 是从日志中解析得到的数据
plot_rtt_and_ratio_over_time(times, rtts, compressed_ratios)
plot_rtt_vs_ratio(rtts, compressed_ratios)