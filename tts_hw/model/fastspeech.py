from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from tts_hw.base.base_model import BaseModel
from tts_hw.model.modules import PositionalEncoding, FFTransformer
from tts_hw.model.duration_modeling import regulate_len, DurationPredictor
from tts_hw.utils import get_mask_from_lengths, get_mask_from_padding


class FastSpeech(BaseModel):

    def __init__(
            self,
            vocab_size,
            n_mel_channels,
            emb_dim,
            enc_num_layers,
            enc_hidden,
            enc_kernel,
            enc_num_heads,
            dec_num_layers,
            dec_hidden,
            dec_kernel,
            dec_num_heads,
            duration_kernel,
            duration_filter_size,
            dropout=0.1,
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.encoder = FFTransformer(enc_num_layers, emb_dim, enc_num_heads, enc_hidden, enc_kernel, dropout=dropout)
        self.decoder = FFTransformer(dec_num_layers, emb_dim, dec_num_heads, dec_hidden, dec_kernel, dropout=dropout)

        self.duration_pred = DurationPredictor(emb_dim, duration_kernel, duration_filter_size, dropout)

        self.proj = nn.Linear(emb_dim, n_mel_channels)

    def forward(self, text_encoded: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        max_duration = kwargs.get("max_duration", 75)

        x = self.pos_enc(self.emb(text_encoded))

        mask = get_mask_from_padding(text_encoded)
        enc_embs = self.encoder(x, mask=mask)

        log_durations = self.duration_pred(enc_embs, mask)
        durations = torch.clamp(torch.exp(log_durations) - 1, 0, max_duration)
        upsampled, lens = regulate_len(enc_embs, durations)
        upsampled = self.pos_enc(upsampled)

        mask = get_mask_from_lengths(lens, upsampled.size(1))
        dec_embs = self.decoder(upsampled, mask=mask)
        out = self.proj(dec_embs)
        return out, lens, log_durations

    def infer(self):
        pass
