import torch
from torch.utils.data import Dataset
import torchaudio


class ControlDataset(Dataset):

    def __init__(self):
        super(ControlDataset, self).__init__()
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.sents = [
            "A defibrillator is a device that gives a high energy "
            "electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance "
            "function defined between probability distributions on a given metric space",
        ]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        transcript = self.sents[index]
        tokens, token_lengths = self._tokenizer(transcript)
        return transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
