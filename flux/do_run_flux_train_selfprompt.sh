#!/bin/bash


python sd-scripts/flux_train.py \
      --mixed_precision bf16 \
      --pretrained_model_name_or_path "/data/team/cv/aigc_code/fluxgym/models/unet/flux1-dev.sft" \
      --clip_l "/data/team/cv/aigc_code/fluxgym/models/clip/clip_l.safetensors" \
      --t5xxl "/data/team/cv/aigc_code/fluxgym/models/clip/t5xxl_fp16.safetensors" \
      --ae "/data/team/cv/aigc_code/fluxgym/models/vae/ae.sft" \
      --seed 42 \
      --mixed_precision bf16 \
      --optimizer_type adamw8bit \
      --learning_rate 4e-4 \
      --resolution 1024 \
      --max_train_epochs 10 \
      --dataset_config 'dataset_Mbokeji.toml' \
      --output_dir "/data/private/zongwx1/outputs" \
      --output_name "flux_Mbokeji_v2_100_dim128" \


