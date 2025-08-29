import numpy as np
import os
import sys
import datasets
import argparse
from typing import Tuple
import transformers
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import matplotlib as plt
import random
from tqdm import tqdm
import pandas as pd
from torch.optim import lr_scheduler
from typing import Callable, Dict, List, Tuple, Union
import csv
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')
#first data exploration script for datamining phase 
parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default="./cache")
parser.add_argument('--model_name', type=str, default="Salesforce/codet5p-770m")
parser.add_argument('--path_checkpoint', type=str, default="./replication_package_Human_Ai_Styl/checkpoint.bin")
args = parser.parse_args()

cache_dir = args.cache_dir
model_name = args.model_name
path_checkpoint = args.path_checkpoint

os.environ["HF_HOME"] = cache_dir
os.environ["HF_TOKEN"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["HF_ASSETS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir





def load_tokenizer(tokenizer_name:str, cache_dir:str)->object:
    """
    Function to load the tokenizer by the model's name

    Args: 
     - tokenizer_name -> the name of the tokenizerto download
     - cache_dir -> directory for caching

     Returns:
     - tokenizer -> returns respectively the model and the tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name,cache_dir=cache_dir)

    return tokenizer


def load_model(model_name:str,cache_dir:str)->object:
    """
     Function for model loading

     Args: 
     - model_name -> the name of the model
     - cache_dir -> directory for caching

     Returns:
     - model,tokenizer -> returns respectively the model and the tokenizer
    """

    print(f'...Loading Human-AI stylometer model ...')

    model_kwargs = {}
    model_kwargs.update(dict( torch_dtype=torch.bfloat16, cache_dir=cache_dir))
    transformers.T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
    model_encoder = transformers.T5EncoderModel.from_pretrained(model_name, **model_kwargs)

    return model_encoder

class stylometer_classifier(torch.nn.Module):
    def __init__(self,pretrained_encoder,dimensionality):
        super(stylometer_classifier, self).__init__()
        self.modelBase = pretrained_encoder
        self.pre_classifier = torch.nn.Linear(dimensionality, 768, dtype=torch.bfloat16)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 1, dtype=torch.bfloat16)




    def forward(self, input_ids, padding_mask):
        output_1 = self.modelBase(input_ids=input_ids, attention_mask=padding_mask)
        hidden_state = output_1[0]
        #Here i take only the cls token representation for further classification
        cls_output = hidden_state[:, 0]
        pooler = self.pre_classifier(cls_output)
        afterActivation = self.activation(pooler)
        pooler_after_act = self.dropout(afterActivation)
        output = self.classifier(pooler_after_act)

        if output>=0.07:
            return {"my_class":"It's a Human!",
                   "prob":output}
        else:
            return {"my_class":"It's an LLM!",
                   "prob":output}


def adapt_model(model:object, dim:int=1024) -> object:
    """
    This function returns the model with a classification head
    """
    newModel = stylometer_classifier(model,dimensionality=dim)

    return newModel




def main():


    DEVICE = "cpu"

    #load tokenizer
    tokenizer = load_tokenizer(model_name,cache_dir)

    #loading model and tokenizer for functional translation
    model = load_model(model_name,cache_dir)

    #adding classification head to the model
    model = adapt_model(model, dim=model.shared.embedding_dim)
    model.to(DEVICE)
    model.load_state_dict(torch.load(path_checkpoint,map_location='cpu'))
    model = model.eval()

    print("Welcome to the Human-AI stylomety tool, insert the code you want to inspect here, \n you can end input with Ctrl+D (linux or mac) or Ctrl+Z and enter for windows, to exit the tool enter Ctl+C:  \n")
    with torch.no_grad():
        try:
            while True:
                print("Welcome to the Human-AI stylomety tool, insert the code you want to inspect here, \n you can end input with Ctrl+D (linux or mac) or Ctrl+Z and enter for windows, to exit the tool enter Ctl+C:  \n")
                print("Enter your code here:")
                user_input = sys.stdin.read().strip()
                tokenized_input = tokenizer([user_input])
                out = model(torch.tensor(tokenized_input.input_ids),torch.tensor(tokenized_input.attention_mask))
                print("\n",out["my_class"],"\n") 
        except KeyboardInterrupt:
            print("\n Shutting down the Human-AI tool...")
if __name__ == '__main__':
    main()
