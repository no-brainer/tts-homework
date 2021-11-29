from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import tts_hw.datasets
from tts_hw.alignment.aligner import GraphemeAligner
from tts_hw.collate_fn.collate import CollatorFn
from tts_hw.featurizer.featurizer import MelSpectrogram
from tts_hw.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, grapheme_aligner: GraphemeAligner, featurizer: MelSpectrogram):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(ds, tts_hw.datasets))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=CollatorFn(aligner=grapheme_aligner, featurizer=featurizer),
            shuffle=shuffle, num_workers=num_workers, batch_sampler=batch_sampler)
        dataloaders[split] = dataloader
    return dataloaders
