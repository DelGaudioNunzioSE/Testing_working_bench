import torch
from pathlib import Path
from interface.methods_testing.CodeT5.sniffer.gptsniffer import CodeT5pClassifier
from interface.methods_testing.CodeT5.sniffer.dataset import CodeDataset
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd



def test_model(test_path:str, model_path:str = './temp/CodeT5/pytorch_model.bin'):
    model = CodeT5pClassifier()                     
    sd = torch.load("pytorch_model.bin", map_location="cuda")
    model.load_state_dict(sd)
    model.to("cuda")


    df = pd.read_csv(test_path)

    test_dataset = CodeDataset(df)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model.eval()
    y_true, y_score = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to('cuda') for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to('cuda')
            logits = model(**inputs)["logits"]
            probs = F.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    return y_true, y_score