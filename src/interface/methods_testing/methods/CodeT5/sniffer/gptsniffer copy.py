"""
this code come from GPTSniffer.ipynb
"""

#!pip install torch transformers pandas
# !pip install Keras-Preprocessing



import warnings
import torch
import torch.nn as nn
from transformers import T5EncoderModel, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
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


#B = batch size = nubers of inputs (codes).
#L = sequence length = token inside de codes.
#H = hidden size = embedding dimention of the token.


import warnings
import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class CodeT5pClassifier(nn.Module):
    def __init__(self, model_name="Salesforce/codet5p-220m",
                 num_labels=2, dropout=0.3, quantize=False, use_lora=False):
        super().__init__()

        if quantize:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.enc = T5EncoderModel.from_pretrained(
                model_name, quantization_config=bnb_cfg, device_map="auto"
            )
            # Abilita gradienti sicuri in 8-bit
            self.enc = prepare_model_for_kbit_training(self.enc)
        else:
            self.enc = T5EncoderModel.from_pretrained(model_name)

        # Applica LoRA solo all'ENCODER (q,k,v,o e feed-forward)
        if use_lora:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q", "k", "v", "o", "wi", "wo"],
                task_type=TaskType.FEATURE_EXTRACTION  # stai estraendo embedding per una clf
            )
            self.enc = get_peft_model(self.enc, lora_cfg)
            # optional: recommended for T5 encoder
            self.enc.gradient_checkpointing_enable()

        H = self.enc.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(H, num_labels)

        # tieni un tokenizer unico e corretto
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        if input_ids is None:
            warnings.warn("input_ids è None")
        if attention_mask is None:
            warnings.warn("attention_mask è None")

        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(out.last_hidden_state)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}




