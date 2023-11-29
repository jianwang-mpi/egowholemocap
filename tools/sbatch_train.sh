#!/usr/bin/env bash

set -x

PARTITION=${1:-gpu20}
JOB_NAME=$2
TIME=${3:-"24:00:00"}
CONFIG=$4
GPUS=${GPUS:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
sbatch -p ${PARTITION} \
    -t ${TIME} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    ${SRUN_ARGS} \
    tools/python_train.sh ${CONFIG} ${PY_ARGS}
