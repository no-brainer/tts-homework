import re

import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, mode, limit=None):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.mode = mode
        self.limit = limit

        full_size = super().__len__()
        self.train_size = int(0.8 * full_size)
        self.test_size = full_size - self.train_size

        self.pattern = re.compile(r"[^a-zA-Z !'(),.:;?\-_]")

    def __len__(self):
        if self.limit is not None:
            return self.limit

        if self.mode == "train":
            return self.train_size
        return self.test_size

    def __getitem__(self, index: int):
        if self.mode != "train":
            index += self.train_size

        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = self.pattern.sub("", transcript)

        tokens, token_length = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_length

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
