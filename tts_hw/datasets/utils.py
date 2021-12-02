from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import tts_hw.datasets
from tts_hw.collate_fn.collate import CollatorFn
from tts_hw.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        dataset = configs.init_obj(params["dataset"], tts_hw.datasets)

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
            dataset, batch_size=bs, collate_fn=CollatorFn(),
            shuffle=shuffle, num_workers=num_workers, batch_sampler=batch_sampler)
        dataloaders[split] = dataloader
    return dataloaders
