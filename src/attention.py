import math
import torch
import torch.nn as nn
class MaskedScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super.__init__()
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, seq_len, device):
        return torch.tril(
            torch.ones(seq_len, seq_len, device = device)
        )
    
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        d_k = Q.size(-1)
        scores = scores / math.sqrt(d_k)
        mask = self.create_mask(Q.size(-2), Q.device)
        scores = scores.masked_fill(mask==0, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, V)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.attention = MaskedScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )
        return x.transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(self, x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        out = self.attention(Q, K, V)
        out = self.merge_heads(out)
        out = self.Wo(out)

        return out

