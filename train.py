import os
import urllib.request

import torch
from torch.utils.data import DataLoader

from src.config import GPTConfig
from src.dataset import GPTDataset
from src.model import MiniGPT
from src.trainer import Trainer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = GPTConfig()

os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# --------------------------------------------------
# Download Tiny Shakespeare
# --------------------------------------------------

data_path = "data/tinyshakespeare.txt"

if not os.path.exists(data_path):

    print("Downloading Tiny Shakespeare...")

    url = (
        "https://raw.githubusercontent.com/karpathy/"
        "char-rnn/master/data/tinyshakespeare/input.txt"
    )

    urllib.request.urlretrieve(url, data_path)

    print("Dataset downloaded.")


# --------------------------------------------------
# Load text
# --------------------------------------------------

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Total characters:", len(text))


# --------------------------------------------------
# Train / Validation split
# --------------------------------------------------

split_index = int(0.9 * len(text))

train_text = text[:split_index]
val_text = text[split_index:]

print("Training characters  :", len(train_text))
print("Validation characters:", len(val_text))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = GPTDataset(
    train_text,
    config.max_seq_len
)

val_dataset = GPTDataset(
    val_text,
    config.max_seq_len
)

print("Training samples  :", len(train_dataset))
print("Validation samples:", len(val_dataset))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = MiniGPT(config)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)


# --------------------------------------------------
# Scheduler
# --------------------------------------------------

total_steps = len(train_loader) * config.epochs

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps
)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()


# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    config=config
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(config.epochs):

    train_loss = trainer.train_one_epoch()
    val_loss = trainer.validate()

    print(
        f"Epoch {epoch + 1}/{config.epochs} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )