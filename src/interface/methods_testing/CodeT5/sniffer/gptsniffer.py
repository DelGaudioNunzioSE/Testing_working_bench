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


class CodeT5pClassifier(nn.Module):
    def __init__(self, model_name="Salesforce/codet5p-220m", num_labels=2, dropout=0.3, quantize=False):
        super().__init__()

        if quantize: # quantize
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.enc = T5EncoderModel.from_pretrained(
                model_name, quantization_config=bnb_cfg, device_map="auto"
            )
        else:
            self.enc = T5EncoderModel.from_pretrained(model_name) # trasformer encoder [B, L, H]

        H = self.enc.config.d_model          # dimension H x token

        self.dropout = nn.Dropout(dropout) 

        self.classifier = nn.Linear(H, num_labels) # in input H x token (no because we will use pooled)




    def forward(self, input_ids=None, attention_mask=None, labels=None):  # input_ids: [B, L], attention_mask: [B, L], labels: [B]
        '''Rembemer to use sofrtmax at the output'''

        if input_ids is None:
            warnings.warn("input_ids: is None")
        if attention_mask is None:
            warnings.warn("input_ids: is None")

        out = self.enc(input_ids=input_ids, attention_mask=attention_mask) # input trasformer
        mask = attention_mask.unsqueeze(-1) # we dont have H dimension inside attention mask -> [B, L, 1]

        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # mean polling only on valid token (not the padding)

        logits = self.classifier(self.dropout(pooled)) # just the drop out

        loss = None
        if labels is not None: # we avaluate loss only if we are in learning fase
            loss = nn.CrossEntropyLoss()(logits, labels) # 
        #else:
            # logits = torch.softmax(logits, dim=-1) !!!!!!!!!!!!!!!!!! i will do it at the output !!!!!!!!!!!!!!!!!
        return {"loss": loss, "logits": logits}
    


    def input_tokenizer(self, code):
        tokenizer =  AutoTokenizer.from_pretrained("/home/N.DELGAUDIO5/hugging/codet5p-220m-local",  use_fast=True)
        inputs = self.tokenizer.encode_plus(
            code,
            padding='max_length',   # sequenza fissa
            max_length=1024,
            truncation=True         # taglia oltre 1024
        )
        return inputs



