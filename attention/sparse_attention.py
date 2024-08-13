# attention/sparse_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, sparse_indices):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply sparse mask
        mask = torch.zeros_like(attn_weights).to(attn_weights.device)
        for b in range(B):
            for h in range(self.n_heads):
                mask[b, h, sparse_indices[b, h, :, 0], sparse_indices[b, h, :, 1]] = 1

        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        output = self.out_proj(attn_output)
        return output

if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2

    x = torch.rand(batch_size, seq_len, d_model)
    sparse_indices = torch.randint(0, seq_len, (batch_size, n_heads, seq_len, 2))
    attention = SparseAttention(d_model, n_heads)
    output = attention(x, sparse_indices)
    print(output)
