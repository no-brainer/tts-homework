import os

import numpy as np
import torch
import torch.nn as nn


class PrecomputedAligner(nn.Module):

    def __init__(self, root):
        super(PrecomputedAligner, self).__init__()
        self.root = root

    @torch.no_grad()
    def forward(self, index, *args, **kwargs):
        durations = []
        for idx in index:
            durations.append(np.load(
                os.path.join(self.root, f"{idx}.npy")
            ))

        return torch.from_numpy(np.stack(durations))
