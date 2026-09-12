import torch
import torch.nn as nn

from src.config import GPTConfig
from src.model import MiniGPT


config = GPTConfig()

model = MiniGPT(config)
print("Embedding std:", model.token_embedding.weight.std().item())

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (2, config.max_seq_len)
)

targets = torch.randint(
    0,
    config.vocab_size,
    (2, config.max_seq_len)
)

# Forward
logits = model(input_ids)
print("Logits min :", logits.min().item())
print("Logits max :", logits.max().item())
print("Logits mean:", logits.mean().item())
print("Logits std :", logits.std().item())

# Loss
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(-1, config.vocab_size),
    targets.view(-1)
)

print("Loss before backward:", loss.item())

# Backward
optimizer.zero_grad()
loss.backward()

print("Backward pass: OK")

# Save one parameter before update
before = model.token_embedding.weight[0, 0].item()

# Update parameters
optimizer.step()

# Check same parameter after update
after = model.token_embedding.weight[0, 0].item()

print("Parameter before:", before)
print("Parameter after :", after)

if before != after:
    print("Optimizer update: OK")
else:
    print("Optimizer update: FAILED")