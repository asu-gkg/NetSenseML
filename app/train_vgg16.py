import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集的本地存储路径
data_path = './data/cifar100'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整 CIFAR-100 图像大小为 VGG16 所需的 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 CIFAR-100 数据集（如果已经下载过了，设置 download=False）
train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型保存路径
model_path = './models/vgg16_cifar100_pretrained.pth'
# 检查保存目录是否存在，如果不存在则创建
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 如果本地有预训练的 VGG16 模型参数，则从本地加载
if os.path.exists(model_path):
    print("Loading VGG16 model from local file...")
    model = models.vgg16(pretrained=False)  # 不再下载预训练权重
    model.load_state_dict(torch.load(model_path))  # 加载本地的权重
else:
    print("Downloading and saving VGG16 pretrained model weights...")
    model = models.vgg16(pretrained=True)  # 下载预训练模型
    torch.save(model.state_dict(), model_path)  # 保存预训练权重到本地

# 替换最后一层分类器以适应 CIFAR-100 (100 类)
model.classifier[6] = nn.Linear(4096, 100)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 定义学习率调度器
num_epochs = 100
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

plt.ion()  # 打开交互模式
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
epoch_list = []
loss_list = []
accuracy_list = []
time_list = []
start_time = time.time()

# 训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0  # 用于累加每个 epoch 的总损失
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
    # 学习率调度器更新
    lr_scheduler.step()

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

    # 每个 epoch 完成后打印损失和准确率
    print(f"Epoch {epoch + 1}/{num_epochs} finished, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

plt.savefig("vgg16_training_plot.png")  # 保存为 PNG 格式

# 保存训练后的模型
torch.save(model.state_dict(), "./models/vgg16_cifar100_trained.pth")

print("VGG16 模型训练完成并保存。")