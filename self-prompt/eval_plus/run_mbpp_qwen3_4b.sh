#!/bin/bash

# 1. 创建结果存放目录
mkdir -p results/mbpp

# 2. 定义变量
MODEL_TYPE="qwen3_4b"   # 对应 generate.py 中 MODEL_MAPPING 的一级 key
MODEL_SIZE="chat"       # 对应 generate.py 中 MODEL_MAPPING 的二级 key
DATASET="mbpp"

export VLLM_N_GPUS=1 


echo "Running generation for $MODEL_TYPE $MODEL_SIZE on $DATASET..."

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

python ./generate.py \
  --model_type $MODEL_TYPE \
  --model_size $MODEL_SIZE \
  --bs 1 \
  --temperature 0.01 \
  --n_samples 1 \
  --resume \
  --greedy \
  --root outputs \
  --dataset $DATASET

# 4. 运行评估
# 注意：这里的路径必须与 generate.py 生成的文件夹名称一致
# generate.py 通常会生成 "model_type + _task_ + task_id" 格式的文件夹
# 如果你没指定 task_id，默认可能是 outputs/humaneval/qwen3_4b_task_None
# 你可以先运行生成，看一眼 outputs 目录下生成的文件夹名叫什么，再填入下面

GEN_OUTPUT_DIR="outputs/${DATASET}/${MODEL_TYPE}_task_None"

echo "Evaluating results from $GEN_OUTPUT_DIR..."

evalplus.evaluate \
  --dataset $DATASET \
  --samples $GEN_OUTPUT_DIR > results/${DATASET}/${MODEL_TYPE}_${MODEL_SIZE}.txt

echo "Done! Results saved to results/${DATASET}/${MODEL_TYPE}_${MODEL_SIZE}.txt"