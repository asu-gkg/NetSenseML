import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT 模型和 tokenizer
model_dir = "./models/bert-base-uncased"  # 使用 BERT base 模型
model_name = "bert-base-uncased"

model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
# 读取 SST-2 数据集文件
data_dir = './data/stanfordSentimentTreebank'

# 加载句子和字典
sentences_df = pd.read_csv(os.path.join(data_dir, 'datasetSentences.txt'), sep='\t')
sentiments_df = pd.read_csv(os.path.join(data_dir, 'sentiment_labels.txt'), sep='|')
dictionary_df = pd.read_csv(os.path.join(data_dir, 'dictionary.txt'), sep='|', names=['phrase', 'phrase_ids'])

# 加载数据集划分信息
split_df = pd.read_csv(os.path.join(data_dir, 'datasetSplit.txt'), sep=',')

# 将句子映射到情感标签
dictionary_df['sentiment_value'] = dictionary_df['phrase_ids'].map(sentiments_df.set_index('phrase ids')['sentiment values'])

# 将句子与标签合并
sentences_df['sentiment_value'] = sentences_df['sentence'].map(dictionary_df.set_index('phrase')['sentiment_value'])

# 定义情感标签二分类（0 = negative, 1 = positive）
sentences_df['label'] = (sentences_df['sentiment_value'] >= 0.5).astype(int)

# 合并划分信息，0为训练集，1为验证集，2为测试集
sentences_df = pd.merge(sentences_df, split_df, left_on='sentence_index', right_on='sentence_index')

# 自定义 Dataset 类，用于加载本地数据
class SST2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]["sentence"]
        label = self.data.iloc[idx]["label"]
        
        # 将句子进行编码
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# 根据数据集划分进行训练集、验证集的构建
train_data = sentences_df[sentences_df['splitset_label'] == 1]
val_data = sentences_df[sentences_df['splitset_label'] == 2]

# 创建自定义数据集
train_dataset = SST2Dataset(train_data, tokenizer)
val_dataset = SST2Dataset(val_data, tokenizer)

# 设置 PyTorch 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 30
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
model.save_pretrained("./bert-finetuned-sst2")
tokenizer.save_pretrained("./bert-finetuned-sst2")

print("BERT SST-2 微调完成并保存。")