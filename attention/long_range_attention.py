# attention/long_range_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LongRangeAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super(LongRangeAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_output = torch.zeros_like(v)

        for t in range(0, T, self.block_size):
            q_block = q[:, :, t:t+self.block_size, :]
            k_block = k[:, :, t:t+self.block_size, :]
            v_block = v[:, :, t:t+self.block_size, :]

            attn_weights = (q_block @ k_block.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)

            attn_output[:, :, t:t+self.block_size, :] = attn_weights @ v_block

        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        output = self.out_proj(attn_output)
        return output

if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 16
    d_model = 8
    n_heads = 2
    block_size = 4

    x = torch.rand(batch_size, seq_len, d_model)
    attention = LongRangeAttention(d_model, n_heads, block_size)
    output = attention(x)
    print(output)
