from src.dataset import GPTDataset


text = """
Machine learning is a field of artificial intelligence.
Deep learning uses neural networks to learn representations
from data. Transformer models are especially powerful for
sequence modelling.
"""

dataset = GPTDataset(
    text=text,
    max_seq_len=8
)

print("Number of samples:", len(dataset))

input_ids, target_ids = dataset[0]

print("Input :", input_ids)
print("Target:", target_ids)
print("Input shape :", input_ids.shape)
print("Target shape:", target_ids.shape)