from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from tts_hw.utils import get_mask_from_lengths


class FastSpeechLoss(nn.Module):

    def __init__(self, duration_pred_coef):
        super(FastSpeechLoss, self).__init__()
        self.duration_pred_coef = duration_pred_coef

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Dict]:
        mel_tgt = kwargs.get("mel_tgt")
        mel_lens = kwargs.get("mel_lengths")
        mel_mask = get_mask_from_lengths(mel_lens, mel_tgt.size(1))

        mel_pred = kwargs.get("mel_pred")
        mel_pred = F.pad(mel_pred, (0, 0, 0, mel_tgt.size(1) - mel_pred.size(1), 0, 0))
        mel_loss = (F.mse_loss(mel_pred, mel_tgt, reduction="none") * mel_mask).sum() / mel_mask.sum()

        dur_tgt = torch.log1p(kwargs.get("dur_tgt"))
        dur_lens = kwargs.get("phoneme_lengths")
        dur_mask = get_mask_from_lengths(dur_lens, dur_tgt.size(1))

        dur_log_pred = kwargs.get("dur_log_pred")
        duration_loss = (F.mse_loss(dur_log_pred, dur_tgt, reduction="none") * dur_mask) / torch.sum(dur_mask)

        loss = mel_loss + self.duration_pred_coef * duration_loss

        results = dict(mel_loss=mel_loss, duration_loss=duration_loss)
        return loss, results
