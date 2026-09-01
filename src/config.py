from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 128
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 1024
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda"