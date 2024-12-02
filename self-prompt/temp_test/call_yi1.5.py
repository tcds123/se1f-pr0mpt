from transformers import AutoModelForCausalLM, AutoTokenizer

# model_path = '/data/team/zongwx1/llm_models/Yi-1.5-9B-Chat'
model_path = 'D:/Users/zongwx1/Downloads/Yi-1.5-9B-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
print(input_ids)
for id in input_ids[0]:
    print([tokenizer.decode(id)])
output_ids = model.generate(input_ids.to(model.device),
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=2048,
                            )
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)

if __name__ == "__main__":
    pass