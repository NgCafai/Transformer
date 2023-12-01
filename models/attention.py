import math
import torch
from torch import nn

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Args:
        query: (batch_size, num_heads, seq_len_q, d_k), given sequence that we focus on
        key: (batch_size, num_heads, seq_len_k, d_k), the sequence to check relevance with query
        value: (batch_size, num_heads, seq_len_v, d_k), seq_len_k == seq_len_v, usually value and key come from the same source
        mask: for encoder, mask is [batch_size, 1, 1, seq_len_k], for decoder, mask is [batch_size, 1, seq_len_q, seq_len_k]
        dropout: nn.Dropout(), optional
    Returns:
        output: (batch_size, num_heads, seq_len_q, d_v), attn: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    # size of scores: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, value), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout_prob=0.1):
        """
        Args:
            h: number of heads
            d_model: dimension of the vector for each token in input and output
            dropout_prob: probability of dropout
        """
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        # corresponding to W^Q, W^K, W^V (matrix for each head are concatenated for efficiency)   
        # and W^O in the paper
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model), seq_len_k == seq_len_v
            mask: 
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1. linear projection for query, key, value
        #    after this step, the shape of each is (batch_size, h, seq_len, d_k)
        query, key, value = [linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for linear, x in zip(self.linears, (query, key, value))]

        # 2. scaled dot product attention
        #    out: (batch_size, h, seq_len_q, d_k)
        out, _  = scaled_dot_product_attention(query, key, value, mask, self.dropout)

        # 3. "Concat" using a view and apply a final linear
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.d_k)
        )
        out = self.linears[-1](out)

        del query, key, value
        return out
