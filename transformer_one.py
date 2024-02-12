# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # imports
#
# practicing implementing a transformer with an encoder, a decoder, and an encoder+decoder  

# %%
import torch 
import math 
# import torch.nn.functional as F 
from torch import nn  

print(torch.__version__)


# %% [markdown]
# # Positional Encoder Class 

# %%
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size, max_seq_len=512):
        """
        embedding_size == d_model (model dimensions)
        """
        super(PositionalEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len 
        
        # positional encoder tensor
        pe = torch.zeros(max_seq_len, embedding_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2, dtype=torch.float) * -(math.log(10000.0)/embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x 


# %%
z = torch.zeros(4, 8)
print(type(z))
z 

# %%
z[:, 0::2] = 1
z[:, 1::2] = 2
z

# %%
z[:, :4]

# %%
torch.arange(0, 12, dtype=torch.float).unsqueeze(1)

# %%
a = torch.arange(0, 8, 2, dtype=torch.float)
print(a)
torch.exp(a)

# %%
math.log(10.)

# %%
2.718 ** 4

# %%
2 ** 3


# %% [markdown]
# # Multi-headed Attention Class 

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        """
        embedding_size: embedding dimension size, model dimensions 
        """
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        
        self.query_linear = nn.Linear(embedding_size, embedding_size)
        self.key_linear = nn.Linear(embedding_size, embedding_size)
        self.value_linear = nn.Linear(embedding_size, embedding_size)
        self.output_linear = nn.Linear(embedding_size, embedding_size)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim) 
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
    
    @staticmethod
    def compute_attention(query, key, mask=None):
        scores = torch.matmul(query, key.permute(1,2,0))
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-1e9"))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        return attention_weights 
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)
        
        attention_weights = self.compute_attention(query, key, mask)
        
        output = torch.matmul(attention_weights, value)
        output = (output
                  .view(batch_size, self.num_heads, -1, self.head_dim)
                  .permute(0, 2, 1, 3)
                  .contiguous()
                  .view(batch_size, -1, self.embedding_size))
        
        return self.output_linear(output)


# %%

# %%
7 // 2


# %% [markdown]
# # Encoder Only Transformer Class

# %%
class FeedForwardSublayer(nn.Module):
    def __init__(self, model_dimensions, dim_between_layers):
        super(FeedForwardSublayer, self).__init__()
        self.fc1 = nn.Linear(model_dimensions, dim_between_layers)
        self.fc2 = nn.Linear(dim_between_layers, model_dimensions)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, model_dimensions, num_heads, dim_between_layers, dropout):
        """
        Args:
            model_dimensions: d_model
            num_heads: 
            dim_between_layers: d_ff
            dropout: 
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dimensions, num_heads)
        self.feed_forward = FeedForwardSublayer(model_dimensions, dim_between_layers)
        self.norm1 = nn.LayerNorm(model_dimensions)
        self.norm2 = nn.LayerNorm(model_dimensions)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask) # ??? 
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x 
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_len):
        """
        Args:
            vocab_size: 
            d_model: model_dimensions
            num_layers: 
            num_heads: 
            d_ff: dim_between_layers
            dropout: 
            max_seq_len: 
        """
        super(TransformerEncoder, self).__init__() 
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(embedding_size=d_model, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(model_dimensions=d_model, num_heads=num_heads, dim_between_layers=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x 
    

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        logits = self.fc(x)
        return torch.nn.functional.log_softmax(logits, dim=-1)

class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        return self.fc(x) 

# %%

# %%
