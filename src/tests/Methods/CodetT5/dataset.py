# Dataset Class
import warnings
import torch
import torch.nn as nn
from transformers import T5EncoderModel, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np
#from timm.optim.lion import Lion
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


class CodeDataset(Dataset):
    def __init__(self, csv_path, text_col="code", label_col="label", tokenizer =  AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", use_fast=True) ):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_path)
        # opzionale: filtra righe valide
        df = df.dropna(subset=[text_col, label_col])
        self.codes = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        code = self.codes[index]
        label = self.labels[index]
        inputs = self.tokenizer.encode_plus(
            code,
            padding='max_length',   # sequenza fissa
            max_length=1024,
            truncation=True         # taglia oltre 1024
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    