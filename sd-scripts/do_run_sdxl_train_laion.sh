#!/bin/bash

MODEL_PATH=/data/team/zongwx1/llm_models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors
TOKENIZER_PATH=./tokenizer
CONFIG_PATH=./sd_xl_base.yaml
DATASET_PATH=./dataset/laion/
OUTPUT_DIR=.
OUTPUT_NAME=sd_xl_ft
CONFIG_FILE=./config.toml
DATASET_CONFIG=./train_dataset_config.toml
DATASET_NAME=laion


python sdxl_train.py \
      --pretrained_model_name_or_path $MODEL_PATH \
      --config_file $DATASET_CONFIG \
      --tokenizer_cache_dir $TOKENIZER_PATH \
      --original_config $CONFIG_PATH \
      --train_data_dir $DATASET_PATH \
      --shuffle_caption \
      --output_dir $OUTPUT_DIR \
      --resolution 512 \
      --output_name $OUTPUT_NAME \
      --train_batch_size 1 \
      --train_text_encoder \
      --caption_extension="txt" \
      --max_train_epochs 10 \
      --dataset_name $DATASET_NAME \
      --seed 5