import logging
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from tts_hw.utils.util import get_mask_from_lengths

logger = logging.getLogger(__name__)


class CollatorFn:

    def __init__(self, aligner, featurizer):
        self.aligner = aligner
        self.featurizer = featurizer

    def __call__(self, instances: List[Tuple]) -> Dict:
        if len(instances[0]) > 3:
            waveform, waveform_length, transcript, tokens, token_lengths = list(
                zip(*instances)
            )
        else:
            waveform, waveform_length = None, None
            transcript, tokens, token_lengths = list(zip(*instances))

        melspec, melspec_length = None, None
        durations = None

        if waveform is not None:
            waveform = pad_sequence([
                waveform_[0] for waveform_ in waveform
            ]).transpose(0, 1)
            waveform_length = torch.cat(waveform_length)

            waveform, waveform_length, melspec, melspec_length, durations = self.featurize_and_align(waveform,
                                                                                                     waveform_length,
                                                                                                     transcript)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return {
            "waveform": waveform,
            "waveform_lengths": waveform_length,
            "text": transcript,
            "text_encoded": tokens,
            "text_encoded_lengths": token_lengths,
            "melspec": melspec,
            "melspec_lengths": melspec_length,
            "durations": durations,
        }

    def featurize_and_align(self, waveforms, waveform_lengths, transcript):
        durations = self.aligner(waveforms, waveform_lengths, transcript)

        full_duration = durations.sum(axis=1)
        durations /= full_duration[:, None]

        waveform_lengths = (waveform_lengths.double() * full_duration).long()
        waveforms *= get_mask_from_lengths(waveform_lengths, waveforms.size(1))

        melspec = self.featurizer(waveforms)
        melspec_length = (waveform_lengths / self.featurizer.hop_length).long()
        melspec = melspec.transpose(-1, -2)

        durations *= melspec_length[:, None]
        
        return waveforms, waveform_lengths, melspec, melspec_length, durations
