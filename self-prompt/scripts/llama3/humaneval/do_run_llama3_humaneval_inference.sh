mkdir -p results/humaneval

MODEL_TYPE=llama3
DATASET=humaneval
TASK=$1

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
  --task $TASK

evalplus.evaluate \
  --dataset $DATASET \
  --samples $SAMPLE_PATH > $RESULT_PATH

