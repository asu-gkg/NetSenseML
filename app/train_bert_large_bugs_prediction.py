import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datasets
import argparse
import logging
import time
import os

def parse():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--log_file', type=str, help='log file path')
    args = parser.parse_args()
    return args

args = parse()


logging.basicConfig(
    filename=args.log_file,  # 输出日志到文件
    filemode='a',  # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 日志级别
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    train_file_path = './data/Bugs-Prediction/embold_train.json'
    with open(train_file_path, 'r') as f:
        train_data = json.load(f)

    # Convert the JSON to a pandas DataFrame
    train_df = pd.DataFrame(train_data)

    # Save as CSV for faster future loading
    train_df.to_csv('./data/Bugs-Prediction/embold_train.csv', index=False)

    # Save as Parquet (even faster than CSV)
    train_df.to_parquet('./data/Bugs-Prediction/embold_train.parquet', index=False)
    print(train_df.columns)
    return train_df

train_df = load_dataset()

# Split data into text and label
# Assuming 'body' contains the text and 'label' contains the target
train_texts = train_df['body'].tolist()  # Adjust based on your column names
train_labels = train_df['label'].tolist()  # Assuming 'label' is the classification label

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Convert texts and labels into Hugging Face Dataset
train_dataset = datasets.Dataset.from_dict({'text': train_texts, 'labels': train_labels})
val_dataset = datasets.Dataset.from_dict({'text': val_texts, 'labels': val_labels})

# Tokenize the entire dataset using batched tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=32)
val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=32)

# Convert to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

config = BertConfig.from_pretrained('bert-large-uncased', num_labels=3)

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained('./models/bert-large-uncased', config=config)
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 30
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 定义准确率计算函数
def compute_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).sum().item()

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    start_time = time.time()
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct_predictions += compute_accuracy(logits, labels)
        total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        progress_bar.set_postfix(loss=f'{avg_loss:.3f}', accuracy=f'{accuracy:.3f}')

    avg_train_loss = total_loss / len(dataloader)
    
    # Validation
    model.eval()
    eval_loss = 0
    eval_accuracy = 0

    for batch in tqdm(val_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            eval_accuracy += (preds == labels).cpu().numpy().mean()

    avg_val_loss = eval_loss / len(val_dataloader)
    avg_val_accuracy = eval_accuracy / len(val_dataloader)
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time  # 每秒处理样本数
    logging.info(f"Epoch {epoch + 1}/{epochs} finished, "
                 f"Test_Loss: {avg_val_loss:.4f}, Test_Accuracy: {avg_val_accuracy:.4f}, "
                 f"Training Throughput: {throughput:.2f} samples/sec")

# Save model
model.save_pretrained('./bert-bug-prediction')
tokenizer.save_pretrained('./bert-bug-prediction')