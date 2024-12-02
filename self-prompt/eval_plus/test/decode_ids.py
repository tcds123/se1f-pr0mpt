import torch
from transformers import AutoTokenizer


MODEL_PATH = '../../chatglm3_6b'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

ids = [1665, 2, 260, 906, 267, 13, 30994, 274, 10516, 31002, 31007, 291, 293, 430, 306, 437, 3387, 30910, 323, 1230, 30930, 30932, 30939, 753, 30962, 1276]
decoded_dict = {}
for id in ids:
    decoded_id = TOKENIZER.decode([id])
    decoded_dict[id] = [decoded_id]

print(decoded_dict)

if __name__ == '__main__':
    pass