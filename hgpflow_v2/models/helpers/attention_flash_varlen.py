import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_qkvpacked_func

class MultiheadFlashAttentionVarLen(nn.Module):
    def __init__(self, 
        embed_dim,
        num_heads,
        bias=True,
        dropout=0.0,
    ):
        super().__init__()

        # Check that the dimension of each heads makes internal sense
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.dropout = dropout

        # Better parallelism for self-attention when using parameters directly
        self.kqv_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.kqv_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kqv_weight)
        if self.bias:
            nn.init.constant_(self.kqv_bias, 0.0)
        self.out_proj.reset_parameters()


    def forward(self, q, culens, maxlen):
        qkv = F.linear(q, self.kqv_weight, self.kqv_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        # Convert to float16 for the flash-varlen backend
        qkv_dtype = qkv.dtype
        qkv = qkv.to(torch.float16) if qkv_dtype != torch.float16 else qkv

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out = flash_attn_varlen_qkvpacked_func(qkv, culens, maxlen, dropout)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Convert back to the original dtype
        if qkv_dtype != torch.float16:
            a_out = a_out.to(qkv_dtype)

        # Mix with final linear layer
        return self.out_proj(a_out)
