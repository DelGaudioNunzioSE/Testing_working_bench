from Methods.CodetT5.gptsniffer import CodeT5pClassifier
from Methods.CodetT5.dataset import CodeDataset
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, Trainer, TrainingArguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
if device == 'cpu':
    warnings.warn("Using cpu because cuda not available")




train_dataset = CodeDataset('./src/tests/Dataset/CodeMirage_train.csv')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the testing dataset and dataloader
val_dataset = CodeDataset('./src/tests/Dataset/CodeMirage_val.csv')
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the training arguments and the trainer
training_args = TrainingArguments(
    output_dir='./src/tests/Output/CodeT5',
    num_train_epochs=12,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    optim='adamw_torch',
    learning_rate=5e-5,
    save_total_limit=2,
    report_to="none",
    save_strategy="epoch",      # o "no"/"steps"
    save_safetensors=False,     # <— chiave
    # metric_for_best_model='f1',
    # report_to='wandb',
    # push_to_hub=True,
    # hub_strategy='every_save',
    # hub_model_id=repository_id,
    # hub_token=HfFolder.get_token(),
)


model = CodeT5pClassifier().to(device)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataloader,
)

# Train the model with the pre-defined parameters
trainer.train()