import torch.nn as nn
from torch.nn import functional as F
import torch
block_size = 512
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
N = 6
d_model = 512
n_heads = 8
#25000#tokenizer.get_vocab_size()
d_ff = 2048
learning_rate = 3e-4
max_iters = 7000
eval_interval = 500

import json
with open('tokenizer/char_to_index.json','r') as f:
  char_to_index = json.load(f)
with open('tokenizer/index_to_char.json','r') as f:
  index_to_char = json.load(f)

encode = lambda seq: [char_to_index[i] for i in seq]
decode = lambda seq: [index_to_char[str(i)] for i in seq]
vocab_size = len(char_to_index)
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):# inp --> (B, C),  out --> (B, C, d_model))
        return self.embedding(x)
    
class PositionalEmbeddings(nn.Module):
    def __init__(self, block_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.pos_embedding = nn.Embedding(block_size, d_model)
    def forward(self, x):# inp --> (C), out --> (C, d_model))
        return self.pos_embedding(x)
    
class Attention(nn.Module):
    def __init__(self, d_model, d, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d = d
        self.Q = nn.Linear(d_model, d)
        self.K = nn.Linear(d_model, d)
        self.V = nn.Linear(d_model, d)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x): #inp --> (64, 256, d_model)
        q = self.Q(x) #(64, 256, d)
        k = self.K(x) #(64, 256, d)
        v = self.V(x) #(64, 256, d)
        T = x.shape[1]
        weights = q@k.transpose(-2,-1)*k.shape[-1]**(-0.5) #(64, 256, 256) 
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1) #(64, 256, 256)
        out = weights @ v
        return out #(64, 256, d)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.heads = nn.ModuleList([Attention(d_model, d_model//n_heads) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * (d_model//n_heads) , d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.2):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        out = self.W1(x)
        out = F.relu(out)
        out = self.dropout(self.W2(out))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.multi_attention = MultiHeadAttention(d_model, n_heads)
        self.ffb = FeedForwardBlock(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.multi_attention(x)
        out1 = x + self.ln1(out)
        out2 = self.ffb(out1)
        final_out = out1 + self.ln2(out2)
        return final_out      

class nanoGPT(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, vocab_size, block_size, N):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.inp_embed = InputEmbeddings(vocab_size, d_model)
        self.pos_embed = PositionalEmbeddings(block_size, d_model)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(N)])
        self.proj = nn.Linear(d_model, vocab_size)
            

    def forward(self, x, targets = None):
        x = self.inp_embed(x)
        block_size = x.shape[1]
        x = x + self.pos_embed(torch.arange(block_size).to(device))
        for block in self.decoder_blocks:
            x = block(x)
        logits = self.proj(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
def generate(gpt, idx, max_new_tokens=1500):
    for ix in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = gpt(idx_cond)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        res = "".join(decode([i.item() for i in idx[0]]))
        if '<endofsong>' in res:
          break
    return res