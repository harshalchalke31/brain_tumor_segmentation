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

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(embed_dim=hidden_dim,num_heads=num_heads,key_dim=hidden_dim,dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim,hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self,x):
        skip1 = x
        x = self.norm(x)
        attn_out = self.attn(x,x,x)
        x = skip1 + attn_out

        skip2 = x
        x = self.norm(x)
        mlp_out = self.mlp(x)
        x = skip2 + mlp_out

        return x
    

