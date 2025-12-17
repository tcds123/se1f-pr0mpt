import sys
import os
import json

import evalplus.data.mbpp as mbpp_module

def load_local_mbpp_plus(mini=False, noextreme=False, version="default"):
    # 这里填写您服务器上真实存在的 MbppPlus.jsonl 路径
    # 根据您之前提供的代码，路径应该是这个：
    local_path = "/data/zhuldz/self-prompt/self-prompt/data/MbppPlus.jsonl"
    
    print(f"🔥 [离线模式] 正在强制加载本地数据集: {local_path}")
    
    if not os.path.exists(local_path):
        print(f"❌ 错误: 找不到本地文件 {local_path}")
        sys.exit(1)

    dataset = {}
    try:
        # 直接读取 jsonl 文件，不走 wget 下载
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                # 确保 task_id 是 key
                dataset[item['task_id']] = item
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        sys.exit(1)
        
    print(f"✅ 成功离线加载 {len(dataset)} 条数据。")
    return dataset

# 将 evalplus 原本的下载/加载函数替换为我们的本地加载函数
mbpp_module.get_mbpp_plus = load_local_mbpp_plus
# -----------------------------------------------------------------------------

# 导入主评估逻辑
from evalplus.evaluate import evaluate
from fire import Fire

if __name__ == "__main__":
    Fire(evaluate)