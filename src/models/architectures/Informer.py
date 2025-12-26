import torch
import torch.nn as nn


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, attn_mask=None):
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_model // self.n_heads)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_model // self.n_heads)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_model // self.n_heads)


class InformerAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.prob_sparse_attention = ProbSparseAttention(d_model, n_heads, dropout)
        
    def forward(self, q, k, v, attn_mask=None):
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_model // self.n_heads)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_model // self.n_heads)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_model // self.n_heads)
        attn_output, attn_weights = self.prob_sparse_attention(q, k, v, attn_mask)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.d_model)
        return attn_output, attn_weights
