import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.random.manual_seed(0)
model_id = "/data/team/zongwx1/llm_models/Phi-3-small-8k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    trust_remote_code=True,
)
assert torch.cuda.is_available(), "This model needs a GPU to run ..."
device = torch.cuda.current_device()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
]
_messages = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)
print(_messages)
print('-' * 30)
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
print(input_ids)
print('-' * 30)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output)
print('-' * 30)
print(output[0]['generated_text'])
