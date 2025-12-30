import json
import os

# 替换成您的两个文件的实际路径
FILE_BASE = "/data/zhuldz/self-prompt/outputs/lcb/qwen3_4b_instruct_baseline/output_codegeneration_output_eval.json"
FILE_EXP  = "/data/zhuldz/self-prompt/outputs/lcb/qwen3_4b_instruct_10/output_codegeneration_output_eval.json"

def load_scores(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return {}
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 1. 如果最外层是列表，取第一个元素
    if isinstance(data, list):
        if len(data) > 0:
            data = data[0]
        else:
            return {}
            
    # 2. 进入 'detail' 字段 (如果存在)
    if "detail" in data:
        data = data["detail"]
        
    # 3. 进入 'pass@1' 字段
    if "pass@1" in data:
        return data["pass@1"]
    
    print(f"⚠️ 警告: 在 {filepath} 中未找到正确的分数结构")
    return {}

print("正在加载数据...")
scores_base = load_scores(FILE_BASE)
scores_exp  = load_scores(FILE_EXP)

# 确保加载成功
if not scores_base or not scores_exp:
    print("❌ 数据加载失败，请检查路径")
    exit()

# 统计
wins = []   # exp 对 (1.0), base 错 (0.0)
losses = [] # exp 错 (0.0), base 对 (1.0)
draws = []  # 结果一样
diff_list = [] # 记录所有分数不同的情况

# 遍历所有题号
all_keys = set(scores_base.keys()) | set(scores_exp.keys())

for q_id in all_keys:
    # 确保两边都有这道题
    if q_id not in scores_base or q_id not in scores_exp:
        continue
    
    s_b = scores_base[q_id]
    s_e = scores_exp[q_id]
    
    if s_e > s_b:
        wins.append(q_id)
        diff_list.append((q_id, s_b, s_e, "Win"))
    elif s_e < s_b:
        losses.append(q_id)
        diff_list.append((q_id, s_b, s_e, "Loss"))
    else:
        draws.append(q_id)

print(f"\n============== 对比分析 ==============")
print(f"🟢 进步 (Wins):   {len(wins)} 道题 (Prompt 修正了错误)")
print(f"🔴 退步 (Losses): {len(losses)} 道题 (Prompt 导致了错误)")
print(f"⚪ 持平 (Draws):  {len(draws)} 道题")
print(f"------------------------------------")
print(f"Base 总分: {sum(scores_base.values())}")
print(f"Exp  总分: {sum(scores_exp.values())}")
print(f"====================================")

if len(wins) > 0:
    print(f"\n✅ 进步示例 (前5个): {wins[:5]}")
    
if len(losses) > 0:
    print(f"\n❌ 退步示例 (前5个): {losses[:5]}")
    print("建议检查这些题目的 Output，看看模型是不是因为格式问题或超时导致判错。")