#!/bin/bash
mkdir -p results/mbpp

get_value() {
    key="$1"
    if [ "$key" = "1" ]; then
        echo 1
    fi
    if [ "$key" = "2" ]; then
        echo 2
    fi
    if [ "$key" = "3" ]; then
        echo 3
    fi
    if [ "$key" = "4" ]; then
        echo 5
    fi
    if [ "$key" = "5" ]; then
        echo 10
    fi
    if [ "$key" = "6" ]; then
        echo 15
    fi
    if [ "$key" = "7" ]; then
        echo 20
    fi
}

MODEL_TYPE=gemma2
DATASET=mbpp
BASE_TASK=1

for key in 5 6 7
do
  TASK=$(($BASE_TASK + $key))
  echo "TASK $TASK"
  SYS_PROMPT_INDEX=$(get_value $key)
  echo "SYS_PROMPT_INDEX $SYS_PROMPT_INDEX"
  SAMPLE_PATH=outputs/${DATASET}/${MODEL_TYPE}_task_${TASK}
  RESULT_PATH=results/${DATASET}/${MODEL_TYPE}_task_${TASK}.txt

  VLLM_N_GPUS=8 python ./eval_plus/generate.py \
    --model_type $MODEL_TYPE \
    --model_size chat \
    --bs 1 \
    --temperature 0.05 \
    --n_samples 1 \
    --resume \
    --greedy \
    --root outputs \
    --dataset $DATASET \
    --task $TASK \
    --sys_prompt_index $SYS_PROMPT_INDEX

  evalplus.evaluate \
    --dataset $DATASET \
    --samples $SAMPLE_PATH > $RESULT_PATH
done
