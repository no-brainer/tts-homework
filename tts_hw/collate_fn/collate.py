import logging
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


class CollatorFn:

    def __call__(self, instances: List[Tuple]) -> Dict:
        if len(instances[0]) > 3:
            index, waveform, waveform_length, transcript, tokens, token_lengths = list(zip(*instances))
        else:
            waveform, waveform_length = None, None
            index, transcript, tokens, token_lengths = list(zip(*instances))

        if waveform is not None:
            waveform = pad_sequence([
                waveform_[0] for waveform_ in waveform
            ]).transpose(0, 1)
            waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return {
            "index": index,
            "waveform": waveform,
            "waveform_lengths": waveform_length,
            "text": transcript,
            "text_encoded": tokens,
            "text_encoded_lengths": token_lengths,
        }
