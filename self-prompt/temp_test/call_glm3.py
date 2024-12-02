from transformers import AutoTokenizer, AutoModel
import torch
import platform
import os
import json


# torch.cuda.set_device(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/data/team/zongwx1/llm_models/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device=DEVICE)

model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def get_stream_chat_response(query):
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = query
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=0.8,
                                                                    temperature=0.8, do_sample=False, num_beams=1,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
                # break
        if query:
            break
        print("")


def get_response(user_input=None, my_embeds=None):
    history = [{"system": "You are a helpful assistant called ChatGPT."},
               {"assistant": "你好，"}]
    response, history = model.chat(tokenizer, user_input, temperature=0.05,
                             history=[],
                             top_p=0.3, max_length=1024, do_sample=False, my_embeds=my_embeds)
    print(history)
    return response


def get_stream_response_one_step(user_input, my_embeds=None):
    past_key_values, history = None, []
    stream_chat_func = model.stream_chat(tokenizer, user_input, history=history, top_p=0.8,
                                                                    temperature=0.8, do_sample=False, num_beams=1,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True, my_embeds=my_embeds)
    response = ''
    while response in ['', ' ', '\n', '\r', '▁']:
        response, history, past_key_values = next(stream_chat_func)
    #     print(tokenizer.encode(response))
    #     print(response)
    # print("-------------")
    probs = torch.load('./pt_file/probs.pt')
    stream_chat_func.close()
    torch.cuda.empty_cache()
    return response, probs


def get_jsonl_prompt_response(jsonl_path, dict_name):
    save_path = '../my_training_embedding_demo/formatted_data/glm_gen.jsonl'
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    sys_prompt = '使用中文并简洁地回答问题。'
    with open(save_path, 'w', encoding='utf-8') as f:
        for d in data:
            user_prompt = d[dict_name]
            prompt = sys_prompt + user_prompt
            response = get_response(prompt)
            dict = {'prompt': user_prompt, 'response': response}
            f.write(json.dumps(dict) + '\n')


if __name__ == '__main__':
    prompt = '你好'
    response = get_response(prompt)
    print(response)
    # jsonl_path = '../my_training_embedding_demo/formatted_data/code.jsonl'
    # get_jsonl_prompt_response(jsonl_path, 'prompt')
    