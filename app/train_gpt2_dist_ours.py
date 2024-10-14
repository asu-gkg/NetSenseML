import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import logging
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from ccl.self_comm import adaptive_bbr_comm_hook, adaptive_sparsify_comm_hook
from datasets import load_from_disk

# 解析命令行参数
def parse():
    parser = argparse.ArgumentParser(description='PyTorch GPT-2 Training with DistributedDataParallel')
    parser.add_argument('--world_size', type=int, help='Number of processes participating in the job')
    parser.add_argument('--rank', type=int, help='Rank of the current process')
    parser.add_argument('--dist_url', type=str, help='URL to connect to for distributed training')
    parser.add_argument('--log_file')
    args = parser.parse_args()
    return args

# 加载或保存数据和模型
def load_and_cache_model(model_dir, local_model_dir):
    if not os.path.exists(local_model_dir):
        print(f"本地模型文件夹不存在，从 {model_dir} 加载并保存模型到本地...")
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model.save_pretrained(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)
    else:
        print(f"从本地文件夹 {local_model_dir} 加载模型...")
        model = GPT2LMHeadModel.from_pretrained(local_model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(local_model_dir)
    return model, tokenizer

def load_and_cache_dataset(dataset_dir, local_dataset_dir):
    if not os.path.exists(local_dataset_dir):
        print(f"本地数据集不存在，从 {dataset_dir} 加载并保存数据到本地...")
        dataset = load_dataset(dataset_dir)
        dataset.save_to_disk(local_dataset_dir)
    else:
        print(f"从本地文件夹 {local_dataset_dir} 加载数据集...")
        dataset = load_from_disk(local_dataset_dir)  # 使用 load_from_disk 加载本地数据集
    return dataset


# 分布式初始化
def setup_distributed(rank, world_size, dist_url):
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)

# 清理分布式环境
def cleanup_distributed():
    dist.destroy_process_group()

args = parse()
world_size = args.world_size
rank = args.rank
dist_url = args.dist_url
log_file = args.log_file

logging.basicConfig(filename=log_file, level=logging.INFO)

# 初始化分布式进程组
setup_distributed(rank, world_size, dist_url)

# 加载模型和数据
model_dir = "/mnt/nfs/models/gpt2-offline"
local_model_dir = "./local_model"
model, tokenizer = load_and_cache_model(model_dir, local_model_dir)
# 这里是必须的
tokenizer.pad_token = tokenizer.eos_token

dataset_dir = "/mnt/nfs/openwebtext"
local_dataset_dir = "./local_openwebtext"
dataset = load_and_cache_dataset(dataset_dir, local_dataset_dir)
train_dataset = dataset["train"]

# 数据预处理    
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"], num_proc=8)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# 模型使用 DDP
device = torch.device(f'cuda:0')
model = model.to(device)
model = DDP(model, device_ids=[rank])
model.register_comm_hook(None, adaptive_bbr_comm_hook)

train_sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
train_dataloader = DataLoader(tokenized_dataset, batch_size=32, sampler=train_sampler)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 训练循环，统计每分钟的 perplexity
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    train_sampler.set_epoch(epoch)
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    
    start_time = time.time()  # 记录开始时间
    last_logged_time = start_time
    
    for step, batch in enumerate(progress_bar):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss_so_far = total_loss / (step + 1)
        perplexity = math.exp(avg_loss_so_far)

        progress_bar.set_postfix(loss=loss.item(), perplexity=perplexity)

        # 每分钟记录一次 perplexity 到日志
        current_time = time.time()
        if current_time - last_logged_time >= 60*10:  # 每 10min 记录一次
            logging.info(f'Epoch {epoch + 1}, Step {step}, Perplexity: {perplexity:.4f}')
            last_logged_time = current_time

    avg_loss = total_loss / len(train_dataloader)
    logging.info(f"Epoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}, Perplexity: {math.exp(avg_loss):.4f}")

# 只在主进程保存模型
if rank == 0:
    model.save_pretrained("./gpt2-finetuned-openwebtext")
    tokenizer.save_pretrained("./gpt2-finetuned-openwebtext")
    print("GPT-2 微调完成并保存。")

# 清理分布式环境
cleanup_distributed()