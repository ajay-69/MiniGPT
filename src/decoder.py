import torch
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim),
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from attention import MaskedMultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, x):
        attn_output = self.attention( self.norm1(x) )
        x = x + attn_output
        feedforward_output = self.ffn(self.norm2(x))
        x = x + feedforward_output
        return x

   
