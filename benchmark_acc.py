import re
import matplotlib.pyplot as plt
import datetime


# 正则表达式提取时间、loss
pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Epoch \d+/\d+, loss: ([\d\.]+)"

times = []
losses = []

# 提取数据
for match in re.finditer(pattern, log_data):
    time_str = match.group(1)
    loss = float(match.group(2))
    
    # 解析时间
    time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
    
    times.append(time)
    losses.append(loss)

# 绘制损失图
plt.figure(figsize=(12, 6))
plt.plot(times, losses, 'b-', label='Loss')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.savefig('./results/vgg16_loss_over_time.png')  # 保存图像
