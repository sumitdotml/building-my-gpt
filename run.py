import torch
import torch.nn as nn
import tiktoken

from model import GPTModel
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = 'another day of waking'
txt2 = 'up with a privilege'

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(46893023)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)

print(f"\nInput batch:\n{batch}\n\nOutput shape:\n{out.shape}\n\nOut:\n{out}")
