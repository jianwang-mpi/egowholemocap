#!/usr/bin/env bash

CONFIG=$1
PY_ARGS=${@:2}

python -u tools/test.py ${CONFIG} ${PY_ARGS}
