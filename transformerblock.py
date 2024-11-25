import torch
import torch.nn as nn
from multihead import MultiHeadAttention
from layernorm import LayerNorm
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.feed_forward=FeedForward(cfg)
        
        # 2 layer normalizations
        self.norm1=LayerNorm(cfg["emb_dim"]) # for attn block
        self.norm2=LayerNorm(cfg["emb_dim"]) # for feed forward block
        
        # drop shortcut
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut connection for attn block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x