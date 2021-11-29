import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, mode):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.mode = mode

    def __len__(self):
        full_size = super().__len__()
        train_size = int(0.8 * full_size)
        if self.mode == "train":
            return train_size
        return full_size - train_size

    def __getitem__(self, index: int):
        if self.mode != "train":
            index += int(0.8 * super().__len__())

        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
