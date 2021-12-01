from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tts_hw.model.utils import get_same_padding


def regulate_len(hidden_states: Tensor, durations: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor]:
    # durations is B x L, hidden_states is B x L x F
    repetitions = torch.round(durations.float() * alpha).long()  # B x L
    frame_lengths = repetitions.sum(dim=1)

    out = torch.zeros(
        hidden_states.size(0), frame_lengths.max(), hidden_states.size(2),
        device=hidden_states.device
    )
    for i, (phoneme_seq, reps) in enumerate(zip(hidden_states, repetitions)):
        out[i, :reps.sum()] = torch.repeat_interleave(phoneme_seq, reps, dim=0)

    return out, frame_lengths


class TransposedConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(TransposedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x.transpose(-2, -1)).transpose(-2, -1)


class DurationPredictor(nn.Module):

    def __init__(self, enc_hidden, kernel_size, filter_size, dropout=0.1):
        super(DurationPredictor, self).__init__()

        padding = get_same_padding(kernel_size)

        self.net = nn.Sequential(
            TransposedConv1d(enc_hidden, filter_size, kernel_size, padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            TransposedConv1d(filter_size, filter_size, kernel_size, padding),
            nn.ReLU(inplace=True),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
            nn.Linear(filter_size, 1)
        )

    def forward(self, x, mask=None):
        # predicts log duration
        # x.shape == (B, L, F)
        x = self.net(x).squeeze(-1)
        if mask is not None:
            x.masked_fill_(~mask, 0.)
        return x
