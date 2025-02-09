import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,key_dim,num_heads,dropout_rate=0.0):
        super().__init__()
        self.num_heads=num_heads
        self.key_dim = key_dim #dimension of each head
        self.embed_dim = embed_dim
        self.total_key_dim = self.num_heads*self.key_dim

        # linear projections for query key values
        self.W_q = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)
        self.W_k = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)
        self.W_v = nn.Linear(in_features=embed_dim,out_features=self.total_key_dim)

        # final projection back to embed dimension
        self.out_proj = nn.Linear(in_features=self.total_key_dim,out_features=embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,query,key,value):
        # q, k, v are expected to be in shape (B,N,embed_dim)
        B,N, _ = query.shape

        # project inputs to multi heads
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # reshape and transpose = (B, num_heads, N, key_dim)
        q = q.view(B, self.num_heads, N, self.key_dim).transpose(1,2)
        k = k.view(B, self.num_heads, N, self.key_dim).transpose(1,2)
        v = v.view(B, self.num_heads, N, self.key_dim).transpose(1,2)

        # compute scaled dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / (self.key_dim**0.5)
        attn_weights = torch.softmax(attn_scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,v) # (B, num_heads, N, key_dim)

        # combine heads
        attn_output = attn_output.transpose(1,2).contiguous().view(B, N, self.total_key_dim) #(B, N, key_dim)
        output = self.out_proj(attn_output)  # (B, N, embed_dim)
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()