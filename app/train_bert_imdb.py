import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT 模型和 tokenizer
model_name = "models/bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

tokenizer = BertTokenizer.from_pretrained(model_name)

# 读取 IMDb 数据集（CSV 文件）
data_path = './data/IMDB Dataset.csv'
df = pd.read_csv(data_path)

# 检查数据格式，确保有 'review' 和 'sentiment' 列
print(df.head())  # 检查数据头几行，确保正确读取

# 将情感标签转换为 0（negative）和 1（positive）
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# 自定义 Dataset 类，用于加载 IMDb 数据
class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]["review"]
        label = self.data.iloc[idx]["label"]
        
        # 将句子进行编码
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# 划分训练集和验证集 (80% 训练集, 20% 验证集)
train_size = 0.8
train_data = df.sample(frac=train_size, random_state=42)
val_data = df.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

# 创建自定义数据集
train_dataset = IMDbDataset(train_data, tokenizer)
val_dataset = IMDbDataset(val_data, tokenizer)

# 设置 PyTorch 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 定义准确率计算函数
def compute_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).sum().item()

# 训练和验证函数
def train_epoch(model, dataloader, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        correct_predictions += compute_accuracy(logits, labels)
        total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    return avg_loss, accuracy

def evaluate_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            correct_predictions += compute_accuracy(logits, labels)
            total_samples += labels.size(0)

            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    return avg_loss, accuracy

# 训练和验证循环
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, lr_scheduler)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    
    val_loss, val_acc = evaluate_epoch(model, val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# 保存微调后的模型
model.save_pretrained("./bert-finetuned-imdb")
tokenizer.save_pretrained("./bert-finetuned-imdb")

print("BERT IMDb 微调完成并保存。")