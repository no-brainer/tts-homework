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
        self.mel_silence = -11.5129251

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Dict]:
        mel_tgt = kwargs.get("melspec")
        mel_lens = kwargs.get("melspec_pred_lengths")
        mel_mask = get_mask_from_lengths(mel_lens, mel_pred.size(1)).unsqueeze(2)

        mel_mask = mel_mask.to(mel_tgt.device)

        mel_pred = kwargs.get("melspec_preds")
        mel_tgt = F.pad(mel_tgt, (0, 0, 0, mel_pred.size(1) - mel_tgt.size(1), 0, 0), value=self.mel_silence)

        mel_loss = (F.mse_loss(mel_pred, mel_tgt, reduction="none") * mel_mask).sum() / mel_mask.sum()

        dur_tgt = torch.log1p(kwargs.get("durations"))
        dur_lens = kwargs.get("text_encoded_lengths")
        dur_mask = get_mask_from_lengths(dur_lens, dur_tgt.size(1)).to(dur_tgt.device)

        dur_log_pred = kwargs.get("log_durations_pred")
        duration_loss = (F.mse_loss(dur_log_pred, dur_tgt, reduction="none") * dur_mask).sum() / torch.sum(dur_mask)

        loss = mel_loss + self.duration_pred_coef * duration_loss

        results = dict(mel_loss=mel_loss, duration_loss=duration_loss)
        return loss, results
