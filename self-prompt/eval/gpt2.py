import torch
from load_dataset import load_csv
from transformers import pipeline


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

model_path = '/data/team/zongwx1/llm_models/gpt-2-large'
device = "cuda"  # the device to load the model onto
generator = pipeline('text-classification', model=model_path, device=device)

labels = ["positive", "negative"]
verbalizer = {
    "positive": "a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films.",
    "negative": "apparently reassembled from the cutting-room floor of any given daytime soap ."
}


def get_gpt2_response(prompt):
    results = generator(prompt, max_length=256)

    label = results[0]['label'][-1]

    return int(label)


def main(dataset_path):
    dataset_list = load_csv(dataset_path)
    sum_count = 0
    acc_count = 0
    for data in dataset_list:
        prompt = data[1]
        # sys_prompt = "It was "
        # prompt += sys_prompt
        sys_prompt = ""
        for label in labels:
            sys_prompt += f"{verbalizer[label]} It was {label}"
        prompt = sys_prompt + prompt + "It was "
        label = int(data[0])
        res = get_gpt2_response(prompt)
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
