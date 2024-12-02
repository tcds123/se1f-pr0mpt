from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "/data/team/zongwx1/llm_models/gemma-2-9b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,)

chat = [
    {"role": "user", "content": "Write a hello world program."},
]
system_prompt = "You are a helpful assistant."
prompt = "Write a hello world program."
prompt = f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```python"

# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs.to(model.device),
    max_new_tokens=2048,
)
response = tokenizer.decode(outputs[0])

def gemma2_res_postprocess(response):
    len_num = len('<start_of_turn>model')
    index = response.find('<start_of_turn>model')
    response = response[index+len_num:]

    index = response.find('<end_of_turn>')
    response = response[:index]
    return response

response = gemma2_res_postprocess(response)

print(response)


