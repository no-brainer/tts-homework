import math
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from tts_hw.model.multihead_attention import MultiHeadAttention
from tts_hw.model.utils import get_same_padding


class PosConvFF(nn.Module):

    def __init__(self, d_model: int, d_inner: int, kernel_size: Tuple[int, int], dropout: float = 0.1):
        super(PosConvFF, self).__init__()

        padding = get_same_padding(kernel_size)

        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_inner, d_model, kernel_size, padding=padding),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv(x.transpose(1, 2))
        output = output.transpose(1, 2)
        output += x
        return self.layer_norm(output)


class FFTBlock(nn.Module):

    def __init__(self, d_model: int, num_head: int, d_inner: int, kernel_size: Tuple[int, int],
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super(FFTBlock, self).__init__()

        self.attn = MultiHeadAttention(d_model, num_head, dropout=attn_dropout)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.pos_conv = PosConvFF(d_model, d_inner, kernel_size, dropout)

    def forward(self, x: Tensor,
                padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        out, attn_scores = self.attn(x, x, x, mask=attn_mask)
        out = self.ln(self.dropout(out) + x)
        out *= padding_mask

        out = self.pos_conv(out)
        out *= padding_mask

        return out, attn_scores


class FFTransformer(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, hidden, kernel, dropout=0.1):
        super(FFTransformer, self).__init__()

        self.fft_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_fft_blocks.append(FFTBlock(d_model, num_heads, hidden, kernel, dropout=dropout))

    def forward(self, x, mask=None):
        # mask size is B x L x F
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        for layer in self.fft_blocks:
            x, _ = layer(x, mask, attn_mask)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
