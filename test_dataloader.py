import torch
from torch.utils.data import DataLoader

from src.dataset import GPTDataset


text = """
Machine learning is a field of artificial intelligence.
Deep learning uses neural networks to learn representations
from data. Transformer models are especially powerful for
sequence modelling. Language models learn to predict the
next token from previous tokens.
"""


dataset = GPTDataset(
    text=text,
    max_seq_len=8
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


input_ids, target_ids = next(iter(loader))

print("Input shape :", input_ids.shape)
print("Target shape:", target_ids.shape)

print("Input:")
print(input_ids)

print("Target:")
print(target_ids)