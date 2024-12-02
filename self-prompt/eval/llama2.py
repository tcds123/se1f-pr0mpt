from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
from load_dataset import load_csv


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

model_name = "/data/team/zongwx1/llm_models/llama-2-7b-hf"
device = "cuda"  # the device to load the model onto

config = AutoConfig.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_llama2_response(prompt):
    inputs = tokenizer(prompt, truncation=True, return_tensors="pt").to(device)
    if hasattr(inputs, 'token_type_ids'):
        del inputs['token_type_ids']

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1).tolist()[0]
    return predictions


def main(dataset_path):
    dataset_list = load_csv(dataset_path)
    sum_count = 0
    acc_count = 0
    for data in dataset_list:
        prompt = data[1]
        sys_prompt = "It was "
        prompt += sys_prompt
        label = int(data[0])
        res = get_llama2_response(prompt)
        sum_count += 1
        print(label, res, end="  ")
        if label == res:
            acc_count += 1
            print("correct")
        else:
            print("wrong")
    print(acc_count / sum_count)


if __name__ == "__main__":
    dataset_path = './eval/data/SST-2/test.tsv'
    main(dataset_path)
