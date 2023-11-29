#!/usr/bin/env bash

CONFIG=$1
PY_ARGS=${@:2}

echo ${CONFIG}

python -u tools/train.py ${CONFIG} ${PY_ARGS}
