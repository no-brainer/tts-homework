from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_head: int,
                 kdim: Optional[int] = None, vdim: Optional[int] = None, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_head = num_head
        self.d_model = d_model
        self.d_k = (d_model if kdim is None else kdim) // num_head
        self.d_v = (d_model if vdim is None else vdim) // num_head

        self.proj_q = nn.Linear(d_model, num_head * self.d_k)
        self.proj_k = nn.Linear(d_model, num_head * self.d_k)
        self.proj_v = nn.Linear(d_model, num_head * self.d_v)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_head * self.d_v, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        q = torch.stack(torch.split(self.proj_q(q), self.d_k, dim=-1), dim=1)  # B x n_heads x L_t x d_k
        k = torch.stack(torch.split(self.proj_k(k), self.d_k, dim=-1), dim=1)  # B x n_heads x L_s x d_k
        v = torch.stack(torch.split(self.proj_v(v), self.d_v, dim=-1), dim=1)  # B x n_heads x L_s x d_v

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # B x n_heads x L_t x L_s
        if mask is not None:  # mask shape B x L_t x L_s
            attn_scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))

        attn_scores = self.dropout(F.softmax(attn_scores, dim=-1))
        out = torch.matmul(attn_scores, v)  # B x n_heads x L_t x d_v
        out = torch.cat(torch.split(out, 1, dim=1), dim=-1).squeeze(1)  # B x L x (d_v * n_heads)
        out = self.fc(out)

        attn_scores = attn_scores.sum(dim=1)  # B x L_t x L_s

        return out, attn_scores
