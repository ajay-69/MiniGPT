import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import GPTConfig
from src.dataset import GPTDataset
from src.model import MiniGPT


# -------------------------
# Configuration
# -------------------------

config = GPTConfig()


# -------------------------
# Training text
# -------------------------

text = """
Machine learning is a field of artificial intelligence.
Deep learning uses neural networks to learn representations
from data. Transformer models are especially powerful for
sequence modelling. Language models learn to predict the
next token from previous tokens.

A neural network consists of layers of mathematical
operations that transform an input into an output.
Training a neural network involves calculating a loss,
computing gradients through backpropagation, and updating
the parameters using an optimization algorithm.

Transformers use self attention to allow tokens to interact
with other tokens in a sequence. In a causal language model,
a token can only attend to previous tokens and itself.
This prevents information from future tokens from leaking
into the prediction process.

The model receives a sequence of token IDs and converts
them into dense vector representations using an embedding
layer. Positional information is added so that the model
can distinguish different positions in the sequence.

The representations then pass through multiple Transformer
blocks. Each block contains layer normalization, causal
self attention, residual connections, and a feed forward
network. Finally, a language modeling head produces logits
for every token in the vocabulary.

During training, the model predicts the next token at every
position. Cross entropy loss measures how different these
predictions are from the correct target tokens. Gradients
are calculated using backpropagation and the optimizer
updates the model parameters.
"""


# -------------------------
# Dataset and DataLoader
# -------------------------

dataset = GPTDataset(
    text=text,
    max_seq_len=config.max_seq_len
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# -------------------------
# Model
# -------------------------

model = MiniGPT(config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

# -------------------------
# Get one batch
# -------------------------

input_ids, targets = next(iter(loader))
logits = model(input_ids)

print("Input shape :", input_ids.shape)
print("Target shape:", targets.shape)
print("Logits shape:", logits.shape)

# -------------------------
# Forward pass
# -------------------------



# -------------------------
# Loss
# -------------------------
criterion = nn.CrossEntropyLoss()
loss = criterion(
    logits.view(-1, config.vocab_size),
    targets.view(-1)
)
print("Loss before update:", loss.item())

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
optimizer.step()
print("Training step: OK")