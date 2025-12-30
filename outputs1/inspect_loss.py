import json
import os

# ================= ⚙️ 配置区域 =================
# 请填入您的文件路径
FILE_BASE_OUT  = "/data/zhuldz/self-prompt/outputs1/lcb/qwen3_4b_instruct_baseline/output.json"
FILE_BASE_EVAL = "/data/zhuldz/self-prompt/outputs1/lcb/qwen3_4b_instruct_baseline/output_codegeneration_output_eval_all.json"

FILE_EXP_OUT   = "/data/zhuldz/self-prompt/outputs1/lcb/qwen3_4b_instruct_10/output.json"
FILE_EXP_EVAL  = "/data/zhuldz/self-prompt/outputs1/lcb/qwen3_4b_instruct_10/output_codegeneration_output_eval_all.json"
# ==============================================

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_scores(data):
    """从 eval 文件中提取 {question_id: score} 字典"""
    # 剥洋葱逻辑：处理可能存在的嵌套结构
    if isinstance(data, list) and len(data) > 0: data = data[0]
    if "detail" in data: data = data["detail"]
    if "pass@1" in data: return data["pass@1"]
    return data # 假设本身就是分数像

def index_output(data_list):
    """将 output list 转换为以 question_id 为 Key 的字典"""
    indexed = {}
    if not isinstance(data_list, list):
        print("⚠️ Warning: Output file content is not a list!")
        return {}
    
    for item in data_list:
        # 强制转换为字符串，确保 key 类型统一
        qid = str(item.get("question_id", ""))
        indexed[qid] = item
    return indexed

print("📥 正在加载并索引数据...")

# 1. 加载分数 (Eval)
raw_base_eval = load_json(FILE_BASE_EVAL)
raw_exp_eval  = load_json(FILE_EXP_EVAL)
scores_base = get_scores(raw_base_eval)
scores_exp  = get_scores(raw_exp_eval)

# 2. 加载代码内容 (Output) 并建立索引
raw_base_out = load_json(FILE_BASE_OUT)
raw_exp_out  = load_json(FILE_EXP_OUT)
code_map_base = index_output(raw_base_out)
code_map_exp  = index_output(raw_exp_out)

print(f"✅ 数据加载完成。Base 题目数: {len(scores_base)}, Exp 题目数: {len(scores_exp)}")

# 3. 寻找退步的题目 (Losses)
losses = [] # (qid, base_score, exp_score)

all_keys = set(scores_base.keys()) | set(scores_exp.keys())

for qid in all_keys:
    # 确保两边都有分
    if qid not in scores_base or qid not in scores_exp: continue
    
    s_b = float(scores_base[qid])
    s_e = float(scores_exp[qid])
    
    # 记录退步：Base=1.0 (对), Exp=0.0 (错)
    if s_b > 0.9 and s_e < 0.1:
        losses.append(qid)

print(f"\n🔴 发现 {len(losses)} 道【退步】题目 (Base对 -> Exp错)")

# 4. 深入分析前 3 个退步案例
for i, qid in enumerate(losses[:3]):
    print(f"\n{'='*20} 🕵️ 案例分析 {i+1}/{min(len(losses), 3)}: ID [{qid}] {'='*20}")
    
    # 获取代码
    item_base = code_map_base.get(qid)
    item_exp  = code_map_exp.get(qid)
    
    if not item_base or not item_exp:
        print("❌ 无法在 Output 文件中找到该 ID 的代码，请检查文件对应关系。")
        continue
        
    c_base = item_base['code_list'][0]
    c_exp  = item_exp['code_list'][0]
    
    print(f"✅ [Baseline (正确)]")
    print(f"长度: {len(c_base)} chars")
    print(f"内容摘要:\n{c_base[:300]}...") 
    
    print(f"\n❌ [Experiment (错误)]")
    print(f"长度: {len(c_exp)} chars")
    print(f"内容摘要:\n{c_exp[:300]}...") 
    
    print("\n🔍 自动诊断:")
    if "```" in c_exp and "def " not in c_exp.split("```")[0]:
        print("👉 格式崩坏：Exp 代码包含了 Markdown 标记但提取失败，或包含了解释性文字。")
    elif len(c_exp) > 2000 and len(c_base) < 500:
        print("👉 废话过多/死循环：Exp 代码极长，可能导致了超时 (Timeout)。")
    elif c_exp.strip() == "":
        print("👉 生成为空：Exp 没有生成任何有效内容。")
    else:
        print("👉 逻辑错误：格式看起来没问题，可能是算法写错了，需要人工细看。")

    print("-" * 60)