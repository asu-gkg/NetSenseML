import logging
import torch
import tqdm
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

device = torch.device("cuda")

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

# 定义学习率调度器


epoch_list = []
loss_list = []
accuracy_list = []
time_list = []


# 训练循环
def load_resnet18_model(hook):
    # 定义模型保存路径
    model_path = './models/resnet18_cifar100_pretrained.pth'
    # 检查保存目录是否存在，如果不存在则创建
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 如果本地有预训练的 resnet18 模型参数，则从本地加载
    if os.path.exists(model_path):
        print("Loading RESNET18 model from local file...")
        model = models.resnet18(pretrained=False)  # 不再下载预训练权重
        model.load_state_dict(torch.load(model_path))  # 加载本地的权重
    else:
        print("Downloading and saving RESNET18 pretrained model weights...")
        model = models.resnet18(pretrained=True)  # 下载预训练模型
        torch.save(model.state_dict(), model_path)  # 保存预训练权重到本地

    # 加载 ResNet18 预训练模型，并替换最后一层分类器以适应 CIFAR-100 (100 类)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 替换全连接层，适应 CIFAR-100
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[0])

    # 定义损失函数和优化器
    if hook is not None:
        model.register_comm_hook(state=None, hook=hook)
    return model
        
def load_vgg_model(hook):
    model_path = './models/vgg16_cifar100_pretrained.pth'
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

    model = DistributedDataParallel(model, device_ids=None)
    model.register_comm_hook(None, hook=hook)
    return model

    
def train_model(model, train_dataloader, test_dataloader, train_sampler, limited_time, method, model_name, num_epochs = 60):
    model.train()
    time_reach_to_best_test_accuracy = 0
    best_test_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    init_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0  # 用于累加每个 epoch 的总损失
        start_time = time.time()
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")  # tqdm 进度条
        train_sampler.set_epoch(epoch)
        for step, (inputs, labels) in enumerate(progress_bar):
            if limited_time != None and (time.time() - init_time) > limited_time:
                print("达到了限制时间，提前退出")
                return best_test_accuracy, time_reach_to_best_test_accuracy
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

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
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = max(test_accuracy, best_test_accuracy)
            time_reach_to_best_test_accuracy = time.time() - init_time
            
        epoch_time = (time.time() - start_time)
        throughput = total_samples * 8 / epoch_time  # 每秒处理样本数
        lr_scheduler.step()
        # 每个 epoch 完成后打印损失和准确率
        logging.info(f"Method: {method}, "
                     f"Model: {model_name}, "
                    f"Epoch {epoch + 1}/{num_epochs} finished, "
                    f"Test_Loss: {test_loss:.4f}, Test_Accuracy: {test_accuracy:.4f}, "
                    f"Best Accuracy: {test_accuracy:.4f}, "
                    f"Training Throughput: {throughput:.2f} samples/sec")
        print(f"Epoch {epoch + 1}/{num_epochs} finished, "
                    f"Test_Loss: {test_loss:.4f}, Test_Accuracy: {test_accuracy:.4f}, "
                    f"Best Accuracy: {best_test_accuracy:.4f}, "
                    f"Training Throughput: {throughput:.2f} samples/sec")
        
    torch.save(model.state_dict(), "./models/resnet18_cifar100_trained.pth")
    return test_accuracy, time_reach_to_best_test_accuracy