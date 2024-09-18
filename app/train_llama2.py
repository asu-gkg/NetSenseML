from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import os
# peft
from peft import LoraConfig

# Load model
model_dir = "./models/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")

prompt = "你好呀, 你是谁？"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:0')

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

print("\033[31m" + gen_text + "\033[0m")


lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
model.set_adapter("adapter_1")



messages = [
    {"role": "user", "content": "Give me a random number."},
]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

system_message = """You are Llama, an AI assistant created by Philipp to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

def create_conversation(sample):
    if sample["messages"][0]["role"] == "system":
        return sample
    else:
        sample["messages"] = [{"role": "system", "content": system_message}] + sample["messages"]
        return sample
    


from datasets import load_from_disk
dataset_directory = './dataset'

dataset = load_from_disk(dataset_directory)

columns_to_remove = list(dataset["train"].features)
columns_to_remove.remove("messages")
dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)


dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)


print(dataset)

print(dataset['train'])
print(dataset['train'][0])

def preprocess_function(examples):
    inputs = []
    labels = []
    
    for messages in examples["messages"]:
        input_text = ""
        output_text = ""
        for message in messages:
            if message["role"] == "user":
                input_text += "User: " + message["content"] + "\n"
            elif message["role"] == "assistant":
                if output_text == "":
                    output_text = "Assistant: " + message["content"] + "\n"
                else:
                    output_text += "Assistant: " + message["content"] + "\n"
            elif message["role"] == "system":
                input_text += "System: " + message["content"] + "\n"
        
        inputs.append(input_text)
        labels.append(output_text)

    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
    with tokenizer.as_target_tokenizer():
        model_labels = tokenizer(labels, truncation=True, padding="max_length", max_length=256)

    model_inputs["labels"] = model_labels["input_ids"]
    return model_inputs

# 
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["messages"])

# Split the dataset
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

print(train_dataset)
print(eval_dataset)
print(len(train_dataset[0]['labels']))


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Small Batch Size
    per_device_eval_batch_size=1,   # Small Batch Size
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,  
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,  
)

from tqdm import tqdm
from transformers import AdamW
from torch.utils.data.dataloader import default_collate

from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()

        # update parameter
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.detach()

    def train(self):
        # Initialize progress bar using tqdm
        progress_bar = tqdm(self.get_train_dataloader(), desc="Training")
        total_loss = 0

        # Loop over training epochs
        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0
            for step, batch in enumerate(self.get_train_dataloader()):
                # Perform training step and accumulate loss
                loss = self.training_step(self.model, batch)
                epoch_loss += loss.item()

                # Update the progress bar with loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                progress_bar.update(1)


# Use our  CustomTrainer to train
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(AdamW(model.parameters(), lr=training_args.learning_rate), None)
)

trainer.train()