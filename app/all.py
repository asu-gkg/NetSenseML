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

import sys
import logging

from common import load_resnet18_model, load_vgg_model, train_model
from ccl.self_comm import sparsify_comm_hook, adaptive_sparsify_comm_hook, adaptive_bbr_comm_hook
from prune.unstructured_prune import l1_unstructured_prune_model

def parse():
    parser = argparse.ArgumentParser(description='PyTorch RESNET18 Training')
    parser.add_argument('--world_size', type=int, help='Number of processes participating in the job')
    parser.add_argument('--rank', type=int, help='Rank of the current process')
    parser.add_argument('--dist_url', type=str, help='URL to connect to for distributed')
    parser.add_argument('--log_file', type=str, help='log file path')
    parser.add_argument('--bandwidth', type=str, help='当前带宽')
    parser.add_argument('--scenario', type=str, help='当前场景')
    args = parser.parse_args()
    return args

args = parse()
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
scenario = args.scenario

# dist_url（rendezvous 地址）可以通过环境变量传递或手动指定
dist_url = "tcp://192.168.1.169:8003"


dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)

data_path = './data/cifar100'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整 CIFAR-100 图像大小为 resnet18 所需的 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 CIFAR-100 数据集（如果已经下载过了，设置 download=False）
train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def custom_adaptive_bbr_comm_hook(state, bucket):
    l1_unstructured_prune_model(model, amount=0.5)  # Prune 20% of the parameters before communication
    return adaptive_bbr_comm_hook(state, bucket)


# ResNet18 Model
model = load_resnet18_model(custom_adaptive_bbr_comm_hook)

num_epochs = 3

logging.basicConfig(
    filename=args.log_file + f"_{args.bandwidth}_{args.scenario}",  # 输出日志到文件
    filemode='a',  # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 日志级别
)

def do_scenario(model_name, model):
    logging.info(f"现在训练 {model_name} 我们的方法: ")
    _, time_to_reach_best_accuracy = train_model(model=model, 
                                                    train_dataloader=train_dataloader, 
                                                    test_dataloader=test_dataloader, 
                                                    train_sampler=train_sampler, 
                                                    limited_time=None, 
                                                    method="NetSenseML", 
                                                    num_epochs=num_epochs, 
                                                    model_name=model_name)

    logging.info(f"现在训练 {model_name}  TopK的方法: ")
    model = load_resnet18_model(sparsify_comm_hook)

    train_model(model=model, 
                        train_dataloader=train_dataloader, 
                        test_dataloader=test_dataloader, 
                        train_sampler=train_sampler, 
                        limited_time=time_to_reach_best_accuracy, 
                        method="TopK-0.1", 
                        num_epochs=num_epochs, 
                        model_name=model_name)

    model = load_resnet18_model(None)
    logging.info(f"现在训练 {model_name} AllReduce的方法: ")
    train_model(model=model, 
                        train_dataloader=train_dataloader, 
                        test_dataloader=test_dataloader, 
                        train_sampler=train_sampler, 
                        limited_time=time_to_reach_best_accuracy, 
                        method="AllReduce", 
                        num_epochs=num_epochs, 
                        model_name=model_name)

do_scenario("ResNet18", model)

# model = load_vgg_model(custom_adaptive_bbr_comm_hook)
# do_scenario("VGG", model)