import argparse
import os
from os import PathLike
import re
import textwrap
import json

from model import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_MAPPING = {
    #  Can be either repo's name or /path/to/model
    "glm3": {
        "chat": "/data/team/zongwx1/llm_models/chatglm3-6b"
    },
    "qwen2": {
        "chat": "/data/public/models/base/Qwen/Qwen2-7B-Instruct"
    },
    "qwen3_4b": {
        "chat": "/data/zhuldz/self-prompt/models/Qwen3-4B"
    },
    "qwen3_8b": {
        "chat": "/data/zhuldz/self-prompt/models/Qwen3-8B"
    },
    "llama3": {
        "chat": "/data/team/zongwx1/llm_models/llama3-8b-instruct"
    },
    "gemma2": {
        "chat": "/data/team/zongwx1/llm_models/gemma-2-9b-it"
    },
    "yi1.5": {
        "chat": "/data/team/zongwx1/llm_models/Yi-1.5-9B-Chat"
    },
    "phi3": {
        "chat": "/data/team/zongwx1/llm_models/Phi-3-small-8k-instruct"
    },
    "codeqwen": {
        "base": "Qwen/CodeQwen1.5-7B",
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
        "chat-awq": "Qwen/CodeQwen1.5-7B-Chat-AWQ",
    },
}


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def code_generate(args,
                  workdir: PathLike,
                  model: DecoderBase,
                  id_range=None,
                  specified_data=None):
    with Progress(
        TextColumn(f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus()
        elif args.dataset == "mbpp":
            # # ================= [修改后：手动加载本地 MBPP+ (已解压 .jsonl 版)] =================
            #     import json
            #     import os
                
                # 请确认文件名是否正确，例如 MbppPlus.jsonl
                local_mbpp_path = "/data/zhuldz/self-prompt/self-prompt/data/MbppPlus.jsonl" 
                
                print(f"📂 Loading local MBPP+ dataset from {local_mbpp_path} ...")
                dataset = {}
                
                # 2. 修改读取方式：使用标准 open，模式为 'r' (read)，指定 utf-8 编码
                # 不需要 import gzip，也不需要 gzip.open
                with open(local_mbpp_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 增加一个判断防止空行报错
                        if not line.strip(): 
                            continue
                            
                        item = json.loads(line)
                        # 将 JSONL 转换为 {task_id: item} 的字典格式
                        dataset[item['task_id']] = item
                        
                print(f"✅ Loaded {len(dataset)} tasks locally.")
            # dataset = {}
            # local_file_path = "/data/zhuldz/self-prompt/self-prompt/data/train_full_fixed.jsonl"
            # print(f"Loading local MBPP data DIRECTLY from {local_file_path}...")
                
            # if not os.path.exists(local_file_path):
            #     # 如果文件不存在，这里手动抛出异常以免后面报错
            #     print(f"Error: File not found {local_file_path}")
            #     dataset = {} # 保持为空，或者在这里 return
            # else:
            #     try: 
            #         with open(local_file_path, 'r', encoding='utf-8') as f:
            #             for line in f:
            #                 line = line.strip()
            #                 if not line: continue
            #                 try:
            #                     item = json.loads(line)
                                
            #                     # 获取 task_id
            #                     task_id = item.get('task_id')
            #                     if task_id is None: continue
                                
            #                     # 构造 key
            #                     if isinstance(task_id, int):
            #                         key = f"Mbpp/{task_id}"
            #                     else:
            #                         key = str(task_id) if "Mbpp" in str(task_id) else f"Mbpp/{task_id}"


            #                     # 补全 contract
            #                     if 'contract' not in item:
            #                         item['contract'] = ""

            #                     # 【关键】现在 dataset 是字典了，这行代码才能跑通
            #                     dataset[key] = item
                                
            #                 except json.JSONDecodeError:
            #                     continue
            #     except Exception as e:
            #         print(f"Error loading file: {e}")
            
            # from evalplus.data import get_mbpp_plus
            # dataset = get_mbpp_plus()
            

        if specified_data is not None:
            print(f"Load index from specified_data from {specified_data}")
            with open(specified_data, "r") as f:
                s_d = json.load(f)
            my_dataset = dict()
            if args.dataset == "humaneval":
                for i in s_d["index"]:
                    my_dataset[i] = dataset["HumanEval/" + i]
            elif args.dataset == "mbpp":
                for i in s_d["index"]:
                    my_dataset[i] = dataset["Mbpp/" + i]
            dataset = my_dataset

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            if args.contract_type != "none" and task["contract"] == "":
                continue
            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len([f for f in os.listdir(os.path.join(workdir, p_name)) if f.endswith(".py")])
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            while sidx < args.n_samples:
                model.dataset = args.dataset
                outputs = model.codegen(
                    args.sys_prompt_index,
                    construct_contract_prompt(task["prompt"], args.contract_type, task["contract"]).strip(),
                    #do_sample=not args.greedy,
                    num_samples=args.n_samples - sidx,
                )

                print("ori_outputs: ", outputs)

                if args.dataset == "humaneval":
                    if 'qwen' in model.name.lower():
                        # 直接调用正则提取函数，传入 task["entry_point"]
                        # 注意：这里假设 outputs 是列表，取第一个元素处理后再放回列表
                        outputs = [qwen_humaneval_post_process(outputs[0], task["entry_point"])]
                    else:
                        outputs = postprocess(model, outputs, args.dataset)
                    # outputs = postprocess(model, outputs, args.dataset)
                    # outputs = [qwen_humaneval_post_process(outputs[0], task["entry_point"])]  # qwen2
                elif args.dataset == "mbpp":
                    if 'qwen' in model.name.lower():
                        outputs = [qwen_mbpp_post_process(outputs[0])]
                    else:
                        outputs = postprocess(model, outputs, args.dataset)
                    # outputs = postprocess(model, outputs, args.dataset)
                print("outputs: ", outputs)

                assert outputs, "No outputs from model!"
                for impl in outputs:
                    try:
                        with open(
                            os.path.join(workdir, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            if model.direct_completion:
                                f.write(task["prompt"] + impl)
                            else:
                                f.write(impl)
                    except UnicodeEncodeError:
                        continue
                    sidx += 1


def postprocess(model, outputs, dataset=None):
    model = model.__name__().lower()
    if 'glm' in model:
        outputs = llama_post_process(outputs)
    elif 'qwen' in model:
        outputs = llama_post_process(outputs)
    elif 'llama' in model:
        if dataset == 'humaneval':
            outputs = llama_post_process(outputs)
        elif dataset == 'mbpp':
            # outputs = base_post_process(outputs)
            outputs = llama_post_process(outputs)
    elif 'gemma' in model:
        outputs = base_post_process(outputs)
    elif 'yi' in model:
        outputs = llama_post_process(outputs)

    return outputs

def qwen_humaneval_post_process(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


# def qwen_humaneval_post_process(text, entry_point):
#     """
#     针对 HumanEval 的后处理：
#     必须去掉 'def func_name(...):' 这一行，只保留函数体。
#     """
#     text = text.strip()
    
#     # 1. 清洗 Markdown 标记
#     if "```python" in text:
#         text = text.split("```python")[1]
#     elif "```" in text:
#         text = text.split("```")[1]
#     if "```" in text:
#         text = text.split("```")[0]
        
#     text = text.strip()

#     # 2. 核心逻辑：找到 def 行并截断
#     # 匹配 "def entry_point(...):" 及其后面的换行
#     # re.DOTALL 确保 . 能匹配换行符以外的字符
#     pattern = re.compile(rf"def\s+{entry_point}.*?:\s*\n", re.DOTALL)
#     match = pattern.search(text)
    
#     if match:
#         # 返回 def 之后的所有内容（即函数体）
#         return text[match.end():]
        
#     # 3. 如果没找到 def，可能模型直接输出了 body，或者输出格式不对
#     # 尝试保留原文本，但 HumanEval 要求 body 必须缩进
#     lines = text.split('\n')
#     if lines and not lines[0].startswith((' ', '\t')):
#         return textwrap.indent(text, '    ')
        
#     return text

def qwen_mbpp_post_process(text):
    """
    修复版后处理：保留 import/from 语句，并提取完整代码块
    """
    text = text.strip()
    
    # 1. 优先提取 Markdown 代码块 (Chat 模型)
    if text.startswith("```"):
        if text.startswith("```python"):
            text = text[9:]
        elif text.startswith("```"):
            text = text[3:]
        if "```" in text:
            text = text.split("```")[0]
        return text.strip()
    
    if "```" in text:
        text = text.split("```")[0]
        return text.strip()

    # 2. 针对 Base 模型：提取包含 imports 的代码块
    # 逻辑：找到第一个以 import, from 或 def 开头的行，作为代码起始
    lines = text.split('\n')
    start_index = -1
    
    for i, line in enumerate(lines):
        line_strip = line.strip()
        # 寻找代码的起始特征
        if line_strip.startswith("def ") or \
           line_strip.startswith("import ") or \
           line_strip.startswith("from ") or \
           line_strip.startswith("@"): # 装饰器
            start_index = i
            break
            
    if start_index != -1:
        # 简单的截断策略：从代码开始处截取到文本结束
        # (通常 Base 模型生成完代码后会停止，或者接新的 task_id，或者输出解释)
        # 为了更安全，可以保留原有的 indentation 截断逻辑作为辅助，但在 import 场景下较难通用
        # 这里建议直接返回从 start_index 开始的所有内容，EvalPlus 运行时的容错性通常能处理末尾的杂音
        return "\n".join(lines[start_index:]).strip()

    return text

# def qwen_mbpp_post_process(text):
#     # 正则表达式匹配代码块
#     code_block_pattern = re.compile(
#         r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
#     )
#     code_block = code_block_pattern.search(text)

#     if code_block is not None:
#         return code_block.group(1)

#     # if no code block is found, assume the LM is simply filling the code
#     return textwrap.indent(text, " " * 4)


def base_post_process(outputs):
    outputs = outputs[0]

    python_index = outputs.find('```python')
    index = outputs[python_index:].find('\n')
    outputs = outputs[python_index + index:]

    # remove str after ```
    r_index = outputs.rfind('```')
    outputs = outputs[:r_index]

    # remove str after the last 'return'
    return_index = outputs.rfind('return')
    index = outputs[return_index:].find('\n')
    outputs = outputs[:return_index + index]

    return [outputs]


def llama_post_process(outputs):
    outputs = outputs[0]

    # remove str after ```
    r_index = outputs.rfind('```')
    outputs = outputs[:r_index]

    # remove str after the last 'return'
    return_index = outputs.rfind('return')
    index = outputs[return_index:].find('\n')
    outputs = outputs[:return_index + index]

    return [outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, choices=MODEL_MAPPING.keys())
    parser.add_argument("--model_size", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", type=int)
    parser.add_argument("--sys_prompt_index", type=int, default=-1)
    parser.add_argument("--specified_data", type=str)
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    assert args.model_size in MODEL_MAPPING[args.model_type]

    model_path = MODEL_MAPPING[args.model_type][args.model_size]

    print(f"Running model={args.model_type}, size={args.model_size}")
    print(f"\tLoad from `{model_path}`")

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0.05
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0.05")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model

    # # ----------------------------------------------------------------------
    # # ✅ [新增修改] 读取对应 txt 文件的第二行作为 extra_prompt
    # # ----------------------------------------------------------------------
    # extra_sys_prompt = None
    
    # # 1. 构建文件路径: txt/模型名(小写)/数据集/extra_ids.txt
    # # 注意：这里假设 model_path 的最后一部分是模型名，如 Qwen2-7B-Instruct
    # model_name_dir = os.path.basename(model_path).lower()
    # txt_file_path = os.path.join("txt", model_name_dir, args.dataset, "extra_ids.txt")
    
    # print(f"Checking for system prompt in: {txt_file_path}")

    # if os.path.exists(txt_file_path):
    #     try:
    #         with open(txt_file_path, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
    #             # 检查是否有第二行 (索引为 1)
    #             if len(lines) >= 2:
    #                 # 获取第二行并去除首尾空白符
    #                 extra_sys_prompt = lines[1].strip()
    #                 print(f"✅ Loaded extra system prompt from line 2: {extra_sys_prompt}")
    #             else:
    #                 print(f"⚠️ File found but lines < 2. Content: {lines}")
    #     except Exception as e:
    #         print(f"❌ Error reading prompt file: {e}")
    # else:
    #     print(f"⚠️ Prompt file not found at {txt_file_path}, using default.")

    model = make_model(
        model_type=args.model_type,
        model_size=args.model_size,
        model_path=model_path,
        batch_size=args.bs,
        temperature=args.temperature,
        #my_sys_prompt=extra_sys_prompt,
    )
    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model_type
        + f"_task_{args.task}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"Working dir: {workdir}")

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    print(f"Model cls: {model.__class__}")
    print(f"EOS tokens: {model.eos}")
    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
