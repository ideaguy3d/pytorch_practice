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

# %%
