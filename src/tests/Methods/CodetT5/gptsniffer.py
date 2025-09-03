"""
this code come from GPTSniffer.ipynb
"""

#!pip install torch transformers pandas
# !pip install Keras-Preprocessing



import warnings
import torch
import torch.nn as nn
from transformers import T5EncoderModel, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from Methods.CodetT5.dataset import *
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np
#from timm.optim.lion import Lion
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score




# Set device to GPU if available, otherwise use CPU




# Define the tokenizer and the model ##############################################################

class CodeT5pClassifier(nn.Module):
    def __init__(self, model_name="Salesforce/codet5p-220m", num_labels=2, dropout=0.1):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained(model_name)
        hidden = self.enc.config.d_model          # dimensione embedding dal config
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
    
    def input_tokenizer(self, code):
        tokenizer =  AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", use_fast=True)
        inputs = self.tokenizer.encode_plus(
            code,
            padding='max_length',   # sequenza fissa
            max_length=1024,
            truncation=True         # taglia oltre 1024
        )
        return inputs



