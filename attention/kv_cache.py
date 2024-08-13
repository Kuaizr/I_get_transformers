# attention/kv_cache.py

import torch
import torch.nn as nn

class KeyValueCache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id, key):
        if (layer_id, key) in self.cache:
            return self.cache[(layer_id, key)]
        else:
            return None

    def set(self, layer_id, key, value):
        self.cache[(layer_id, key)] = value

    def clear(self):
        self.cache = {}

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(KVCacheAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_cache = KeyValueCache()

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, layer_id):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        k_cache = self.kv_cache.get(layer_id, 'k')
        v_cache = self.kv_cache.get(layer_id, 'v')

        if k_cache is None or v_cache is None:
            k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            self.kv_cache.set(layer_id, 'k', k)
            self.kv_cache.set(layer_id, 'v', v)
        else:
            k = k_cache
            v = v_cache

        attn_weights = torch.einsum('bhqd, bhkd -> bhqk', q, k) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum('bhqk, bhvd -> bhqd', attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, C)
        output = self.out_proj(attn_output)
        return output

if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 16
    d_model = 8
    n_heads = 2
    layer_id = 0

    x = torch.rand(batch_size, seq_len, d_model)
    attention = KVCacheAttention(d_model, n_heads)
    output = attention(x, layer_id)
    print(output)
