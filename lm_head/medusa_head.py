import torch
import torch.nn as nn
import torch.nn.functional as F

class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads=8, dropout=0.1):
        super(MedusaHead, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Multi-head attention layer
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)

        # Feed-forward neural network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Multi-head attention
        attn_output, _ = self.multi_head_attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm1(hidden_states + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.layer_norm2(attn_output + ffn_output)

        # Output layer
        logits = self.output_layer(ffn_output)

        return logits

if __name__ == "__main__":
    import torch

    # Testing MedusaHead
    batch_size = 2
    seq_length = 10
    hidden_size = 512
    vocab_size = 10000
    num_heads = 8

    medusa_head = MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size, num_heads=num_heads)
    input_tensor = torch.randn(seq_length, batch_size, hidden_size)
    attention_mask = torch.ones(seq_length, seq_length)  # Example attention mask
    output = medusa_head(input_tensor, attention_mask)
    print("MedusaHead output shape:", output.shape)  # Expected: (seq_length, batch_size, vocab_size)
