import torch
import torch.nn as nn
from gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), # linear layer
            GELU(), # gelu activation
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']), # linear layer
        )
    
    def forward(self, x):
        return self.layers(x)