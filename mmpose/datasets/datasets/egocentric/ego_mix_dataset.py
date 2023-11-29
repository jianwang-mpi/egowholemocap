# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.builder import build_dataset
import os
import pickle

@DATASETS.register_module()
class EgoMixDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, configs, partition):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        datasets = [build_dataset(cfg) for cfg in configs]
        self.dataset = ConcatDataset(datasets)
        self.length = max(len(ds) for ds in datasets)
        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(partition, datasets)
        ]
        weights = np.concatenate(weights, axis=0)
        self.sampler = WeightedRandomSampler(weights, 1)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        return self.dataset[idx_new]

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        res_file = os.path.join(res_folder, 'results.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(outputs[:100], f)

        return {'result': 0}
