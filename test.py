import torch
from src.config import GPTConfig
from src.model import MiniGPT

config = GPTConfig()
model = MiniGPT(config)
print("Embedding std:", model.token_embedding.weight.std().item())

input_ids = torch.randint(
    0,
    config.vocab_size,
    (2, config.max_seq_len)
)

logits = model(input_ids)

print("Input shape :", input_ids.shape)
print("Logits shape:", logits.shape)
print("Parameters  :", sum(p.numel() for p in model.parameters()))