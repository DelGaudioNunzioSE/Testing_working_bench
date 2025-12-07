import os
import pandas as pd
from interface.methods_testing.methods.CodeT5.sniffer.gptsniffer import CodeT5pClassifier
from interface.methods_testing.methods.CodeT5.sniffer.dataset import CodeDataset
from interface.methods_testing.methods.CodeT5.sniffer.code_cleaner import comment_remover, newline_remover, import_remover
#from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
#from sniffer.gptsniffer import CodeT5pClassifier
#from sniffer.dataset import CodeDataset
#from sniffer.code_cleaner import comment_remover, newline_remover, import_remover
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments , EarlyStoppingCallback, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset

import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm


def CodeT5_learning(train_path:str = './src/tests/Dataset/CoDET_train.csv', val_path:str = None, use_lora = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if device == 'cpu':
        warnings.warn("Using cpu because cuda not available")


    def preprocess(example):
        code = example["code"]
        if code is not None:
            language = example["language"]
            code = comment_remover(code, language)
            code = import_remover(code, language)
            code = newline_remover(code)
        return {"code": code}



    dataset_train = load_dataset("csv", data_files=train_path)["train"]
    dataset_val = load_dataset("csv", data_files=val_path)["train"] if val_path is not None else None

    dataset_train = dataset_train.filter(lambda ex: isinstance(ex.get("code"), str) and ex["code"] != "", num_proc =4)
    dataset_val = dataset_val.filter(lambda ex: isinstance(ex.get("code"), str) and ex["code"] != "", num_proc =4) if val_path is not None else None

    dataset_train = dataset_train.map(preprocess, num_proc=4)
    dataset_val = dataset_val.map(preprocess, num_proc=4) if val_path is not None else None


    df_t = dataset_train.to_pandas()
    train_dataset = CodeDataset(df_t)

    # Define the testing dataset and dataloader
    df_v = dataset_val.to_pandas() if val_path is not None else None
    val_dataset = CodeDataset(df_v) if val_path is not None else None


    print('Upload model in memory')
    model = CodeT5pClassifier(use_lora=use_lora).to(device)

    path_temp = "./temp/CodeT5"
    os.makedirs(path_temp, exist_ok=True)
    if val_path is not None:
        training_args = TrainingArguments(
            output_dir=path_temp,
            eval_strategy="epoch",          # must match save_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,    # obbligatorio
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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

    else :
        data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)

        training_args = TrainingArguments(
            output_dir=path_temp,
            eval_strategy="no",  
            save_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=False,
            num_train_epochs=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            optim="adamw_torch",
            learning_rate=5e-5,
            save_total_limit=2,
            report_to="none",
            save_safetensors=False,
            do_train=True,
            do_eval=False,
            gradient_accumulation_steps=1,   # increase if you want a larger effective batch
            fp16=False,                      # metti True se sei su GPU Ampere; o bf16=True se supportato
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,   # must expose 'text' and 'label' or already tokenized
            eval_dataset=None,
            data_collator=data_collator,
        )

    # Train the model with the pre-defined parameters
    trainer.train()

    bin_path = os.path.join(path_temp, "pytorch_model.bin")
    torch.save(model.state_dict(), bin_path)

    return  bin_path





if __name__ == "__main__": 
    path_bin = CodeT5_learning(train_path = './CoDET(5).csv', val_path = None, use_lora= True)
    print(path_bin)
