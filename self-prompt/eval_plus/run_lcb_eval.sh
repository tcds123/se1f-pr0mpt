#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# ======================================================
# ⚙️ 配置区域
# ======================================================

# 1. ⚠️ 如果您想跑全量 1055 题，必须设为 false 重新生成！
# 否则它会一直读取您只有 400 个答案的旧文件。
SKIP_INFERENCE="false"
#SKIP_INFERENCE="false"

# 2. 路径配置
MODEL_PATH="/data/zhuldz/self-prompt/models/Qwen3-4B-Instruct-2507" 
#PROMPT_FILE="${PROJECT_ROOT}/self-prompt/txt/qwen3_4b/humaneval/10.txt"
PROMPT_FILE=""
#OUTPUT_FILE="${PROJECT_ROOT}/outputs1/lcb/qwen3_4b_instruct_10/output.json"
OUTPUT_FILE="${PROJECT_ROOT}/outputs1/lcb/qwen3_4b_instruct_baseline/output.json"
# 3. 数据集配置
# 留空代表跑全量 (1055题)
START_DATE=""  
LCB_VERSION="release_latest"

# ======================================================

if [ "$SKIP_INFERENCE" == "false" ]; then
    echo "------------------------------------------------"
    echo "🚀 Step 1: Running Inference (Generating Code)..."
    echo "------------------------------------------------"
    
    # 1. 先定义基础命令 (不包含 sys_prompt_file)
    CMD_INFERENCE="python ${SCRIPT_DIR}/run_lcb_inference.py \
        --model_path $MODEL_PATH \
        --output_file $OUTPUT_FILE \
        --max_new_tokens 2048 \
        --release_version $LCB_VERSION"

    # 2. 只有当 PROMPT_FILE 不为空时，才追加该参数
    # [ -n "$VAR" ] 用于判断变量长度是否大于 0
    if [ -n "$PROMPT_FILE" ]; then
        CMD_INFERENCE="$CMD_INFERENCE --sys_prompt_file $PROMPT_FILE"
    fi

    # 3. 同理，处理 START_DATE
    if [ -n "$START_DATE" ]; then
        CMD_INFERENCE="$CMD_INFERENCE --start_date $START_DATE"
    fi

    # 4. 打印预览一下最终命令 (调试好习惯)
    echo "Executing: $CMD_INFERENCE"

    # 5. 执行
    $CMD_INFERENCE

else
    echo "------------------------------------------------"
    echo "⏭️  Skipping Inference Step (Using existing output)"
    echo "------------------------------------------------"
fi

echo "------------------------------------------------"
echo "📊 Step 2: Evaluating Results"
echo "------------------------------------------------"

# 切换目录
ORIGINAL_DIR=$(pwd)
LCB_ROOT="${SCRIPT_DIR}/LiveCodeBench"

if [ -d "$LCB_ROOT" ]; then
    cd "$LCB_ROOT" || exit 1
else
    echo "❌ Error: LiveCodeBench directory not found at $LCB_ROOT"
    exit 1
fi

export PYTHONPATH=$(pwd):$PYTHONPATH

# 构建 Step 2 命令 (使用单行避免换行符错误)
CMD_EVAL="python -m lcb_runner.runner.custom_evaluator --custom_output_file $OUTPUT_FILE --num_process_evaluate 1 --scenario codegeneration --model qwen-custom"

if [ -n "$START_DATE" ]; then
    CMD_EVAL="$CMD_EVAL --start_date $START_DATE"
fi

# 执行评测
echo "Running: $CMD_EVAL"
$CMD_EVAL

# 切回
cd "$ORIGINAL_DIR"

echo "------------------------------------------------"
echo "✅ Evaluation Done."
echo "------------------------------------------------"

EVAL_FILE="${OUTPUT_FILE/.json/_codegeneration_eval.json}"
if [ -f "$EVAL_FILE" ]; then
    echo "🏆 Final Scores:"
    grep -E "pass@1|easy|medium|hard" "$EVAL_FILE"
fi