import torch 
import torch.nn as nn
from .config import GPTConfig # . means Look in the same package/directory.
from decoder import DecoderBlock
config = GPTConfig()

class MiniGPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(
            config.d_model,
            config.num_heads,
            config.hidden_dim,
            config.dropout)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
# GPT-2 does not use a bias term in the output projection because the shared 
# embedding weights (which we'll add next) already provide enough flexibility.
        self.lm_head.weight = self.token_embedding.weight


    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = (token_embeddings + position_embeddings)
        hidden_states = self.dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits