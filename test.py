import argparse

from scipy.io import wavfile
import torch

from tts_hw.datasets import ControlDataset
from tts_hw.model import FastSpeech
from tts_hw.utils import read_json, prepare_device
from tts_hw.utils.parse_config import ConfigParser


def main(config_path, checkpoint_path):
    config = ConfigParser(read_json(config_path))

    vocoder = Vocoder("./data/waveglow_256channels_universal_v5.pt").eval()
    model = FastSpeech(**config["arch"]["args"])

    device, device_ids = prepare_device(config["n_gpu"])
    vocoder = vocoder.to(device)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    model.eval()
    dataset = ControlDataset()

    for i in range(len(dataset)):
        transcript, tokens, _ = dataset[i]
        tokens = tokens.to(device)
        mel, _ = model.infer(tokens)

        result = vocoder.inference(mel.transpose(-2, -1))

        wavfile.write(
            f"test{i}_{transcript[:32]}.wav", 22050, result.squeeze(0).cpu().numpy()
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test script")
    args.add_argument(
        "config",
        type=str,
        help="config file path",
    )
    args.add_argument(
        "checkpoint",
        type=str,
        help="checkpoint file path",
    )
    args.add_argument(
        "-o",
        "--output",
        default="./audio_samples",
        type=str,
        help="path to saved test audios (default: ./audio_samples)"
    )

    args = args.parse_args()

    main(args["config"], args["checkpoint"])
