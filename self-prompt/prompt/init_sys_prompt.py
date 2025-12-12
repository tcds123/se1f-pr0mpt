import json
import os

import generate_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2Config


def generate_sys_prompt(model_name, model, tokenizer, device, dataset_name):
    min_len, max_len = None, None
    if 'glm3' in model_name:
        gen_function = generate_utils.GLM3Generation(model, tokenizer, device).generate
    elif 'qwen3' in model_name:
        base_gen_func = generate_utils.QWEN3Generation(model, tokenizer, device).generate
        gen_function = lambda prompt, do_sample: base_gen_func(prompt, do_sample=do_sample, enable_thinking=False)
    elif 'qwen2' in model_name:
        gen_function = generate_utils.QWEN2Generation(model, tokenizer, device).generate
    elif 'llama3' in model_name:
        gen_function = generate_utils.LLAMA3Generation(model, tokenizer, device).generate
    elif 'gemma2' in model_name:
        gen_function = generate_utils.GEMMA2Generation(model, tokenizer, device).generate
    elif 'yi1.5' in model_name:
        gen_function = generate_utils.YiGeneration(model, tokenizer, device).generate
    else:
        raise ValueError('Currently glm3, qwen2, llama3, gemma2, yi1.5 model supported...')

    if dataset_name == 'humaneval':
        with open('./formatted_data/humaneval/train.jsonl', "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
            case1 = "\nInput 1: " + train_data[0]["prompt"] + "\nOutput 1: " + train_data[0]["response"]
            case2 = "\nInput 2: " + train_data[1]["prompt"] + "\nOutput 2: " + train_data[1]["response"]
            case = case1 + case2
    elif dataset_name == 'mbpp':
        with open('./formatted_data/mbpp/train.jsonl', "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
            case1 = "Input 1: " + train_data[0]["prompt"] + "\nOutput 1: " + train_data[0]["response"]
            case2 = "\nInput 2: " + train_data[1]["prompt"] + "\nOutput 2: " + train_data[1]["response"]
            case = case1 + case2
    else:
        raise ValueError('Currently humaneval or mbpp dataset supported...')

    prompt = f"""Generate one supplementary text that can help trigger the outputs based on the inputs: 
{case}
The text needs to be general enough to solve other different inputs.
Your response must not contain the inputs and outputs before.
Generate directly, no other useless response.
Here is an example:
"
You are an intelligent programming assistant to produce Python algorithmic solutions.
"
The text is:
"""

    response = get_response(gen_function, prompt, min_len, max_len)

    final_sys_prompt = response
    if not os.path.exists('./system_prompt'):
        os.mkdir('./system_prompt')
    with open('./system_prompt/' + model_name + '_' + dataset_name +'.txt', 'w', encoding='utf-8') as f:
        f.write(final_sys_prompt)
    
    return final_sys_prompt


def get_response(gen_function, prompt, min_len: int = None, max_len: int = None):
    while True:
        response = gen_function(prompt, do_sample=True).strip()
        if min_len is None and max_len is None:
            break
        elif min_len is None and max_len is not None:
            if len(response) < max_len:
                break
        elif min_len is not None and max_len is None:
            if len(response) > min_len:
                break
        else:
            if min_len < len(response) < max_len:
                break

    if ":" in response:
        index = response.find(":")
        response = response[index+1:]

    return response


if __name__ == "__main__":
    # model_path = "/data/team/zongwx1/llm_models/chatglm3-6b"
    # model_path = "/data/team/zongwx1/llm_models/qwen2-7b-instruct"
    #model_path = "/data/public/models/base/Qwen/Qwen2-7B-Instruct"
    #model_path = "/data/zhuldz/self-prompt/models/Qwen3-4B"
    model_path = "/data/zhuldz/self-prompt/models/Qwen3-8B"
    # model_path = "/data/team/zongwx1/llm_models/llama3-8b-instruct"
    # model_path = "/data/team/zongwx1/llm_models/Phi-3-small-8k-instruct"
    # model_path = "/data/team/zongwx1/llm_models/gemma-2-9b-it"
    # model_path = "/data/team/zongwx1/llm_models/Yi-1.5-9B-Chat"
    #model_name = 'glm3'
    #model_name = 'qwen2'
    model_name = 'qwen3_8b'
    # model_name = 'llama3'
    # model_name = 'phi3'
    # model_name = 'gemma2'
    # model_name = 'yi1.5'

    device = "cuda"  # the device to load the model onto

    # print(f"Loading config manually for {model_name} from {model_path}...")
    # try:
    #     config = Qwen2Config.from_pretrained(model_path, trust_remote_code=True)
    # except Exception as e:
    #     # 如果 Qwen2Config 失败（极低概率），尝试回退到 AutoConfig 并打印错误
    #     print(f"Qwen2Config load failed: {e}. Trying AutoConfig...")
    #     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # config.auto_map = {
    #     "AutoModelForCausalLM": "qwen2_model.MyQwen2ForCausalLM",
    #   }
    if model_name != 'phi3':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype="auto",
            trust_remote_code=True,

        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    #dataset_name = 'humaneval'
    dataset_name = 'mbpp'
    sys_prompt = generate_sys_prompt(model_name, model, tokenizer, device, dataset_name)
    print(sys_prompt)
