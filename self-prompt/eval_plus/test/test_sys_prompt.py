import torch
from transformers import AutoTokenizer

model_path = '../../chatglm3_6b/'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

sys_prompt_logits = torch.load("./pt_files/learned_sys_prompt_direct.pt", map_location=torch.device('cpu'))

sys_prompt_ids = torch.argmax(sys_prompt_logits, dim=-1)[0]

print(sys_prompt_ids)

sys_prompt = tokenizer.decode(sys_prompt_ids)

print(sys_prompt)

if __name__ == "__main__":
    pass
