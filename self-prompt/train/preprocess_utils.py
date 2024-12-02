import os.path

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import List
import torch
from abc import ABC, abstractmethod


class BaseDataset(ABC, Dataset):
    def __init__(self, data: List[dict],
                 tokenizer: PreTrainedTokenizer,
                 device,
                 max_source_length: int,
                 max_target_length: int,
                 sys_prompt: str,
                 sys_prompt_len: int,
                 extra_len: int,
                 prompt_template: list or str,
                 sp_token_num: tuple,
                 add_generation_prompt: bool = True,
                 ):
        Dataset.__init__(self)
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.sys_prompt = sys_prompt
        self.sys_prompt_len = sys_prompt_len
        self.extra_len = extra_len
        self.prompt_template = prompt_template
        self.add_generation_prompt = add_generation_prompt
        self.prompt_len = 0
        self.bos = None
        self.eos = None
        self.user_prompt_shell = "Can you complete the following Python function?\n```python\n{prompt}\n```\n"
        self.sp_token_num = sp_token_num

    @abstractmethod
    def __getitem__(self, i) -> dict:
        pass

    def __len__(self):
        return len(self.data)


class GLMDataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        len_sp_token_num = self.sp_token_num[0]
        self.extra_index = self.sys_prompt_len + len_sp_token_num
        self.dataset_name = dataset_name

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]

        history = [
            {"role": "system", "content": self.sys_prompt},
        ]

        inputs = self.tokenizer.build_chat_input(prompt, history=history)
        a_ids = inputs["input_ids"][0].tolist()
        # print(inputs)
        # for id in a_ids[0].tolist():
        #     print([self.tokenizer.decode(id)])
        # assert False

        a_ids = a_ids[:self.extra_index] + [self.tokenizer.pad_token_id] * self.extra_len + a_ids[self.extra_index:]
        # print(a_ids)
        # print(self.tokenizer.decode(torch.LongTensor(a_ids)))
        # assert False

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class Qwen2Dataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        self.bos = '<|im_start|>'
        self.eos = '<|im_end|>'
        self.dataset_name = dataset_name

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]
        assistant = None
        if self.prompt_template is not None:
            self.sys_prompt = self.bos + self.prompt_template[0]["role"] + '\n' + self.prompt_template[0][
                "content"] + self.eos
            if isinstance(self.prompt_template[1]["content"], dict):
                prompt = self.bos + self.prompt_template[1]["role"] + '\n' + self.prompt_template[1]["content"][
                    "prompt"].format(prompt=prompt) + self.eos
            else:
                prompt = self.bos + self.prompt_template[1]["role"] + '\n' + self.prompt_template[1][
                    "content"] + self.eos
            if len(self.prompt_template) > 2:
                assistant = self.bos + self.prompt_template[2]["role"] + '\n' + self.prompt_template[2]["content"]
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False, truncation=True,
                                      max_length=self.max_source_length)

        c_ids = self.tokenizer.encode(text=self.sys_prompt, add_special_tokens=True, truncation=True,
                                      max_length=self.max_source_length)
        a_ids = c_ids[:-1] + [self.tokenizer.pad_token_id] * self.extra_len + [c_ids[-1]] + a_ids

        if assistant is not None:
            d_ids = self.tokenizer.encode(text=assistant, add_special_tokens=False, truncation=True,
                                          max_length=self.max_source_length)
            a_ids += d_ids

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class Llama3Dataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        len_sp_token_num = self.sp_token_num[0]
        self.extra_index = self.sys_prompt_len + len_sp_token_num
        # llama3 model does not hava a pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.assistant = "```python"
        self.assistant = self.tokenizer.encode(self.assistant, add_special_tokens=False)
        self.dataset_name = dataset_name

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]
        # prompt = self.user_prompt_shell.format(prompt=prompt)

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]

        a_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=self.add_generation_prompt,
            return_tensors="pt"
        )
        # print(a_ids)
        # print(self.tokenizer.decode(a_ids[0]))

        a_ids = a_ids.tolist()[0]
        a_ids = a_ids[:self.extra_index] + [self.tokenizer.pad_token_id] * self.extra_len + a_ids[self.extra_index:]
        # print(self.tokenizer.decode(torch.LongTensor(a_ids)))
        # if self.dataset_name == 'humaneval':
        a_ids += self.assistant

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class Gemma2Dataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        len_sp_token_num = self.sp_token_num[0]
        self.extra_index = self.sys_prompt_len + len_sp_token_num
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_name = dataset_name

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]
        prompt = self.prompt_template.format(system_prompt=self.sys_prompt, prompt=prompt)

        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False, truncation=True,
                                      max_length=self.max_source_length)
        # print(a_ids)
        # print(self.tokenizer.decode(a_ids))

        a_ids = a_ids[:self.extra_index] + [self.tokenizer.pad_token_id] * self.extra_len + a_ids[self.extra_index:]
        # print(a_ids)
        # print(self.tokenizer.decode(torch.LongTensor(a_ids)))
        # assert False

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class YiDataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        len_sp_token_num = self.sp_token_num[0]
        self.extra_index = self.sys_prompt_len + len_sp_token_num
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_name = dataset_name

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]
        prompt = self.prompt_template.format(system_prompt=self.sys_prompt, prompt=prompt)

        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False, truncation=True,
                                      max_length=self.max_source_length)
        # print(a_ids)
        # print(self.tokenizer.decode(a_ids))

        a_ids = a_ids[:self.extra_index] + [self.tokenizer.pad_token_id] * self.extra_len + a_ids[self.extra_index:]
        # print(a_ids)
        # print(self.tokenizer.decode(torch.LongTensor(a_ids)))
        # assert False

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class InputOutputDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, max_source_length: int,
                 max_target_length: int, sys_prompt, sys_prompt_len, extra_len, prompt_template):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.extra_len = extra_len
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data
        self.sys_prompt = sys_prompt
        self.sys_prompt_len = sys_prompt_len
        self.prompt_len = 0
        self.prompt_template = prompt_template
        self.bos = '<|im_start|>'
        self.eos = '<|im_end|>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        prompt = data_item["prompt"]
        assistant = None
        if self.prompt_template is not None:
            self.sys_prompt = self.bos + self.prompt_template[0]["role"] + '\n' + self.prompt_template[0]["content"] + self.eos
            if isinstance(self.prompt_template[1]["content"], dict):
                prompt = self.bos + self.prompt_template[1]["role"] + '\n' + self.prompt_template[1]["content"]["prompt"].format(prompt=prompt) + self.eos
            else:
                prompt = self.bos + self.prompt_template[1]["role"] + '\n' + self.prompt_template[1]["content"] + self.eos
            if len(self.prompt_template) > 2:
                assistant = self.bos + self.prompt_template[2]["role"] + '\n' + self.prompt_template[2]["content"]
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False, truncation=True,
                                      max_length=self.max_source_length)
        if isinstance(self.sys_prompt, list):
            c_ids_front = self.tokenizer.encode(text=self.sys_prompt[0], add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
            c_ids_back = self.tokenizer.encode(text=self.sys_prompt[1], add_special_tokens=False, truncation=True,
                                          max_length=self.max_source_length)
            a_ids = c_ids_front + a_ids + c_ids_back + [self.tokenizer.pad_token_id] * self.extra_len
            if not os.path.exists('./pt_file/'):
                os.mkdir('./pt_file')
            if self.prompt_len == 0:
                self.prompt_len = len(a_ids)
            else:
                torch.save(self.prompt_len, './pt_file/prompt_len.pt')
                self.prompt_len = len(a_ids)
        else:
            c_ids = self.tokenizer.encode(text=self.sys_prompt, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
            a_ids = c_ids[:-1] + [self.tokenizer.pad_token_id] * self.extra_len + [c_ids[-1]] + a_ids

            if assistant is not None:
                d_ids = self.tokenizer.encode(text=assistant, add_special_tokens=False, truncation=True,
                                          max_length=self.max_source_length)
                a_ids += d_ids

        b_ids = self.tokenizer.encode(text=data_item["response"], add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }

