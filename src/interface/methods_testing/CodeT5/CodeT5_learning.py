import os
import pandas as pd
from interface.methods_testing.CodeT5.sniffer.gptsniffer import CodeT5pClassifier
from interface.methods_testing.CodeT5.sniffer.dataset import CodeDataset
from interface.methods_testing.CodeT5.sniffer.code_cleaner import comment_remover, newline_remover, import_remover
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments , EarlyStoppingCallback, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset

import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm


def CodeT5_learning(train_path:str = './src/tests/Dataset/CoDET_train.csv', val_path:str = './src/tests/Dataset/CoDET_val.csv'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if device == 'cpu':
        warnings.warn("Using cpu because cuda not available")


    def preprocess(example):
        code = example["code"]
        language = example["language"]
        code = comment_remover(code, language)
        code = import_remover(code, language)
        code = newline_remover(code)
        return {"code": code}



    dataset_train = load_dataset("csv", data_files=train_path)["train"]
    dataset_val = load_dataset("csv", data_files=val_path)["train"]

    dataset_train = dataset_train.map(preprocess, num_proc=4)
    dataset_val = dataset_val.map(preprocess, num_proc=4)


    df_t = dataset_train.to_pandas()
    train_dataset = CodeDataset(df_t)

    # Define the testing dataset and dataloader
    df_v = dataset_val.to_pandas()
    val_dataset = CodeDataset(df_v)


    print('Upload model in memory')
    model = CodeT5pClassifier().to(device)

    path_temp = "./temp/CodeT5"
    os.makedirs(path_temp, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=path_temp,
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
        eval_dataset=val_dataset,       
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=2,      # pacience
            early_stopping_threshold=0.0    
        )]
    )

    # Train the model with the pre-defined parameters
    trainer.train()

    bin_path = os.path.join(path_temp, "pytorch_model.bin")
    torch.save(model.state_dict(), bin_path)

    return  bin_path