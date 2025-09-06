import torch
from pathlib import Path
from Methods.CodetT5.gptsniffer import CodeT5pClassifier
from Methods.CodetT5.dataset import CodeDataset
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd

ckpt = Path("./src/tests/Output/CodeT5/CoDET/checkpoint-3000")                  # cartella che contiene pytorch_model.bin
model = CodeT5pClassifier()                     # stessa classe usata in training
sd = torch.load(ckpt / "pytorch_model.bin", map_location="cuda")
model.load_state_dict(sd)
model.to("cuda").eval()




df1 = pd.read_csv('./src/tests/Dataset/CodeMirage_test.csv')
df2 = pd.read_csv('./src/tests/Dataset/AIGCodeSet_test.csv')

df = pd.concat([df1, df2], ignore_index=True)

test_dataset = CodeDataset(df2)
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

y_true = np.array(y_true); y_score = np.array(y_score)
auc_roc = roc_auc_score(y_true, y_score)

disp = RocCurveDisplay.from_predictions(y_true, y_score)
print(f"{auc_roc:.3f}")
plt.title(f"ROC (AUC={auc_roc:.3f})")
plt.savefig("./src/tests/Output/CodeT5/CoDET/roc.png", dpi=300, bbox_inches="tight")
plt.close()