"""
this code come from GPTSniffer.ipynb
"""

#!pip install torch transformers pandas
# !pip install Keras-Preprocessing



import math
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
import torch.nn.functional as F



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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        self.pad_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"

        self.chunk_size = int(1024)

    def _encode_and_pool(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(out.last_hidden_state)
        # media sui token validi
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        token_counts = mask.sum(dim=1)  # [B,1]
        return pooled, token_counts

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        if input_ids is None:
            raise ValueError("input_ids è None")
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        B, L = input_ids.shape
        cs = self.chunk_size

        if L <= cs:
            pooled, _ = self._encode_and_pool(input_ids, attention_mask)
        else:
            # pad fino a multiplo di cs
            n_chunks = math.ceil(L / cs)
            pad_len = n_chunks * cs - L
            input_ids_p = F.pad(input_ids, (0, pad_len), value=self.pad_id)
            attn_p      = F.pad(attention_mask, (0, pad_len), value=0)

            # [B, n_chunks, cs] -> [B*n_chunks, cs]
            x = input_ids_p.view(B, n_chunks, cs).reshape(B * n_chunks, cs)
            m = attn_p.view(B, n_chunks, cs).reshape(B * n_chunks, cs)

            chunk_pooled, chunk_tokens = self._encode_and_pool(x, m)        # [B*n_chunks,H], [B*n_chunks,1]
            H = chunk_pooled.size(-1)
            chunk_pooled = chunk_pooled.view(B, n_chunks, H)
            chunk_tokens = chunk_tokens.view(B, n_chunks, 1)

            # media pesata sulle sezioni = media globale sui token validi
            pooled = (chunk_pooled * chunk_tokens).sum(dim=1) / chunk_tokens.sum(dim=1).clamp(min=1)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}




