# GPT architecture definition
import math
import torch
import torch.nn as nn
from torch.nn import functional as F # activation, loss function, etc.

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input): # forward pass
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)

class CausalSelfAttention(nn.Module): 
    # Implement multi-head self-attention with causal masking (autoregressive, only look at past tokens)
    def __init__(self, n_embd, n_head, dropout, block_size, bias=True):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by the number of heads."

        # Store hyperparameters
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.block_size = block_size

        # Projection for Query, Key, Value (all with same dimension n_embd) into output dimension 3*n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Projection for output back to the input dimension n_embd
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Check for Flash Attention availability
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create a causal attention mask
            self.register_buffer(
                "bias",
                # Create a lower triangular matrix of ones and reshape it to (1, 1, block_size, block_size)
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # x.size() is (B, T, C)
        # self.c_attn(x) is (B, T, 3*C)
        # dim=2 means split along the embedding dimension(C)
        # q, k, v are (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  
        
        # Reshape for multi-head attention
        # k, q, v are (B, n_head, T, embedding_size per head)
        # Input shape: (32, 60, 384) -> (32, 6, 60, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  

        # Flash Attention or fallback to manual implementation
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, # no additional mask needed for causal attention
                dropout_p=self.dropout if self.training else 0,
                is_causal=True # enable causal masking
            )

        # y has shape (B, n_head, T, embedding_size per head)
        # y.transpose(1, 2) has shape (B, T, n_head, embedding_size per head)
        # reshape back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  

        # self.c_proj(y) projects y back to the input dimension n_embd
        y = self.resid_dropout(self.c_proj(y))
        return y

# SwiGLU used in LLaMa
# Swish-Gated Linear Unit FFN (activation function)
# FFN with SwiGLU activation function
class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        # Expands the embedding dimension by 8/3
        d_ff = int((8/3) * n_embd)
        # Project the input to twice the expanded dimension
        self.fc1 = nn.Linear(n_embd, 2 * d_ff, bias=bias)
        # Project the expanded dimension back to the original dimension
        self.fc2 = nn.Linear(d_ff, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc1(x) # (B, T, 2*d_ff)
        
        x1, x2 = x_proj.chunk(2, dim=-1) # split the input into two halves along the last dimension
        swish = x1 * torch.sigmoid(x1) # Swish activation function x * sigmoid(x)
        
        x = swish * x2 # multiply the activated gate with the value path(x2)
        x = self.fc2(x) # project the output back to the original input dimension
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, bias=True):
        super().__init__()
        
        self.ln_1 = LayerNorm(n_embd, bias=bias) # layer norm for the input
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias) # layer norm for the output of the attention
        self.mlp = SwiGLUFFN(n_embd, dropout, bias=bias) # SwiGLU FFN

    def forward(self, x):
        # pre-normalization
        x = x + self.attn(self.ln_1(x))  # residual connection (x + attention(x))
        x = x + self.mlp(self.ln_2(x))  # residual connection (x + MLP(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True, device=None, padding_token_index=12):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.padding_token_index = padding_token_index

        # nn.ModuleDict stores modules in a dictionary format
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # for each token, create a learnable embedding vector of size n_embd
            drop = nn.Dropout(dropout),
            # nn.ModuleList stores a list of modules
            # a stack of n_layer transformer blocks, 6 layers means 6 transformer blocks
            h = nn.ModuleList([Block(n_embd, n_head, dropout, block_size, bias=bias) for _ in range(n_layer)]), 
            ln_f = LayerNorm(n_embd, bias=bias), # final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False) # projects the final transformer output to the vocab size

        # recursively apply the weight initialization function to all modules in the model
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # For linear layers (fc1, fc2, lm_head):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # initialize the bias to 0
        # For embedding layers (wte)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # idx is the input token indices [3,5,2,10,11,12]
        device = idx.device
        b, t = idx.size() # b is the batch size, t is the sequence length
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # Convert token indices (idx) to token embeddings (tok_emb)
        tok_emb = self.transformer.wte(idx)
        # Apply dropout, only active during training
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h: # 6 transformer blocks
            # forward pass through each transformer block
            x = block(x) 
        x = self.transformer.ln_f(x)
        # Logits are the unnormalized output scores for each token in the vocabulary, with shape (b, t, vocab_size)
        # Linear projection to get logits for each token in the vocabulary
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            # we need the full logits to compute the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), # reshape logits to (b*t, vocab_size)
                                   targets.view(-1), # reshape targets to (b*t)
                                   ignore_index=self.padding_token_index) # ignore the padding token
            
            logits = self.lm_head(x[:, [-1], :]) # select the last position from the sequence dimension for each examples in the batch

        return logits, loss