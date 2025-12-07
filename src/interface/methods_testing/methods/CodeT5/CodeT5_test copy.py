import torch
from pathlib import Path
from interface.methods_testing.methods.CodeT5.sniffer.gptsniffer import CodeT5pClassifier
from interface.methods_testing.methods.CodeT5.sniffer.dataset import CodeDataset
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd
from interface.methods_testing.methods.CodeT5.sniffer.code_cleaner import comment_remover, newline_remover, import_remover
from datasets import load_dataset


def test_model(test_path:str, model_path:str = './model/CodeT5(1).bin'):
    model = CodeT5pClassifier(use_lora = True)                     
    sd = torch.load(model_path, map_location="cuda",  weights_only=False)
    model.load_state_dict(sd if "state_dict" not in sd else sd["state_dict"])
    model.to("cuda")
    print('Uploaded model')
    def preprocess(example):
        code = example["code"]
        language = example["language"]
        code = comment_remover(code, language)
        code = import_remover(code, language)
        code = newline_remover(code)
        return {"code": code}
    print('Preprocessing code')
    dataset_test= load_dataset("csv", data_files=test_path)["train"]
    dataset_test = dataset_test.map(preprocess)
    df = dataset_test.to_pandas()


    test_dataset = CodeDataset(df)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    


    model.eval()
    y_true, y_score = [], []
    print('starting eval')
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to('cuda') for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to('cuda')
            logits = model(**inputs)["logits"]
            probs = F.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    return y_true, y_score





if __name__ == "__main__": 
    path_bin = CodeT5_learning(train_path = './CoDET(5).csv', val_path = None, use_lora= True)
    print(path_bin)