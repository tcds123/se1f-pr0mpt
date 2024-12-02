#! /usr/bin/env bash

set -ex

EXTRA_LEN=0
LR=1e-4
NUM_GPUS=1
MAX_SOURCE_LEN=256
MAX_TARGET_LEN=1024
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=5000
EPOCH=200

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=advertise_gen_pt

MODEL_NAME="glm3"
MODEL_PATH="/data/team/zongwx1/llm_models/chatglm3-6b"
DATASET_NAME="humaneval"
DATASET_PATH=formatted_data/humaneval/train.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${EXTRA_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train/finetune.py \
    --train_format input-output \
    --dataset_name $DATASET_NAME \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --learning_rate $LR \
    --epoch $EPOCH \
    --extra_len $EXTRA_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
