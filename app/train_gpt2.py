import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2TokenizerFast
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import torch

model_dir = "/mnt/nfs/models/gpt2-offline"
device = torch.device("cuda:0")


model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

prompt = "GPT2 is a model developed by OpenAI."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

print("\033[31m" + gen_text + "\033[0m")

# 加载 OpenWebText 数据集
dataset = load_dataset("/mnt/nfs/openwebtext")

train_dataset = dataset["train"]


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 预处理数据集
tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"], num_proc=32)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


# 定义 PyTorch 数据加载器
train_dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0  # 用于累加每个 epoch 的总损失
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")  # tqdm 进度条
    for step, batch in enumerate(progress_bar):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
        loss = outputs.loss

        # 反向传播并更新参数
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()  # 将每个批次的损失累加
        
        # 计算当前 perplexity，基于累积的平均损失
        avg_loss_so_far = total_loss / (step + 1)  # 当前的平均损失
        perplexity = math.exp(avg_loss_so_far)  # 计算 perplexity

        # 在进度条上显示当前损失和 perplexity
        progress_bar.set_postfix(loss=loss.item(), perplexity=perplexity)


    avg_loss = total_loss / len(train_dataloader)  # 计算每个 epoch 的平均损失
    print(f"Epoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}")

# 保存微调后的模型
model.save_pretrained("./gpt2-finetuned-openwebtext")
tokenizer.save_pretrained("./gpt2-finetuned-openwebtext")

print("GPT-2 微调完成并保存。")