#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

GPUS=$(nvidia-smi --list-gpus | wc -l)
CONFIG=$1
PY_ARGS=${@:2}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch ${PY_ARGS}
