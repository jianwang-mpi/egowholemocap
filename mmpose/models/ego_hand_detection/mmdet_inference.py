#  Copyright Jian Wang @ MPI-INF (c) 2023.
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def inference_detector_img_tensor(model, imgs):
    device = next(model.parameters()).device  # model device

    # just get the actual data from DataContainer
    data = {}
    data['img_metas'] = []
    data['img'] = [imgs[i] for i in range(len(imgs))]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    return results

