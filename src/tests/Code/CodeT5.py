import pandas as pd
from Methods.CodetT5.gptsniffer import CodeT5pClassifier
from Methods.CodetT5.dataset import CodeDataset
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments , EarlyStoppingCallback, TrainingArguments, Trainer, EarlyStoppingCallback





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
if device == 'cpu':
    warnings.warn("Using cpu because cuda not available")



df = pd.read_csv('./src/tests/Dataset/CoDET_train.csv', usecols=["cleared_code", "label"])
train_dataset = CodeDataset(df)

# Define the testing dataset and dataloader
df = pd.read_csv('./src/tests/Dataset/CoDET_val.csv', usecols=["cleared_code", "label"])
val_dataset = CodeDataset(df)



model = CodeT5pClassifier().to(device)

training_args = TrainingArguments(
    output_dir="./src/tests/Output/CodeT5/CoDET",
    eval_strategy="epoch",          # deve combaciare con save_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,    # obbligatorio
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    optim="adamw_torch",
    learning_rate=5e-5,
    save_total_limit=2,
    report_to="none",
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,       # necessario
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=2,      # n° valutazioni senza migliorare
        early_stopping_threshold=0.0    # miglioramento minimo richiesto
    )]
)

# Train the model with the pre-defined parameters
trainer.train()