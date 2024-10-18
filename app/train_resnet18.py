import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 ResNet18 预训练模型，并替换最后一层分类器以适应 CIFAR-100 (100 类)
model = models.resnet101(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)  # 替换全连接层，适应 CIFAR-100
model = model.to(device)

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 定义数据集的本地存储路径
data_path = './data/cifar100'

# 加载 CIFAR-100 数据集
train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 记录训练过程的统计数据
epochs = 10
epoch_list = []
loss_list = []
accuracy_list = []
time_list = []
start_time = time.time()

# 初始化 matplotlib 图表
plt.ion()  # 打开交互模式
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")  # tqdm 进度条
    
    for step, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播并更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        # 当前的平均损失和准确率
        avg_loss = total_loss / (step + 1)
        accuracy = correct_predictions / total_samples

        # tqdm 进度条显示
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    # 计算每个 epoch 的平均损失和准确率
    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples
    epoch_time = (time.time() - start_time) / 60  # 时间以分钟计算
    test_loss, test_accuracy = evaluate(model, train_dataloader, criterion)
    # 将数据保存到列表中
    epoch_list.append(epoch + 1)
    loss_list.append(avg_loss)
    accuracy_list.append(accuracy)
    time_list.append(epoch_time)

    # 动态更新图表
    ax[0].cla()  # 清除子图 0
    ax[1].cla()  # 清除子图 1

    # 绘制 epoch vs loss 和 accuracy
    ax[0].plot(epoch_list, loss_list, label='Loss', color='red')
    ax[0].plot(epoch_list, accuracy_list, label='Accuracy', color='blue')
    ax[0].set_title("Epoch vs Loss/Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Value")
    ax[0].legend()

    # 绘制 time vs loss 和 accuracy
    ax[1].plot(time_list, loss_list, label='Loss', color='red')
    ax[1].plot(time_list, accuracy_list, label='Accuracy', color='blue')
    ax[1].set_title("Time (minutes) vs Loss/Accuracy")
    ax[1].set_xlabel("Time (minutes)")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    # 刷新图表
    plt.pause(0.01)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

plt.ioff()  # 关闭交互模式
plt.show()  # 保持图表显示

# 保存图表到文件
plt.savefig("resnet18_training_plot.png")  # 保存为 PNG 格式
print("训练图表已保存为 'resnet18_training_plot.png'")

# 保存最终模型
torch.save(model.state_dict(), "./resnet18_cifar100_trained.pth")
print("ResNet18 模型训练完成并保存。")