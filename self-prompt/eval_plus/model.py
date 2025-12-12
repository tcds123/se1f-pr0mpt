import os
from abc import ABC, abstractmethod
from typing import List
import chardet

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./hf_home")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        direct_completion: bool = True,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        dataset: str = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.direct_completion = direct_completion
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.dataset = dataset
        self.head_prompt = "\nYou should be careful of these tokens: "
        self.use_prompt = True
        self.use_my_sys_prompt = True

        if direct_completion and dataset is not None:
            if dataset.lower() == "humaneval":
                self.eos += ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
            elif dataset.lower() == "mbpp":
                self.eos += ['\n"""', "\nassert"]

    @abstractmethod
    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def get_sys_prompt(self):
        pass

    def __name__(self):
        return self.name

    def get_txt_path(self):
        model_name = self.name.split('/')[-1]
        txt_path = './txt/' + model_name.lower() + '/' + self.dataset + '/'
        return txt_path

    def get_system_prompt(self, i):
        txt_path = self.get_txt_path()
        with open(txt_path + str(i) + '.txt', 'rb') as f:
            data = f.read()
            encoding = chardet.detect(data)['encoding']
            system_prompt = data.decode(encoding)

        return self.head_prompt + system_prompt

    def is_direct_completion(self):
        return self.direct_completion


class GLM(DecoderBase):
    def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
        kwargs["direct_completion"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, device=DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, device=DEVICE)

        # 1. 无论有值没值，先占个坑，防止 AttributeError
        self.my_sys_prompt = my_sys_prompt if my_sys_prompt is not None else ""
        
        # 2. 根据是否有值，来决定开关状态（这完美保留了您的判断逻辑）
        self.use_my_sys_prompt = True if self.my_sys_prompt else False

    def codegen(self, sys_prompt_index: int,prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        print("********** do_sample: ", do_sample)
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        with open('./prompt/system_prompt/glm3_' + self.dataset.lower() + '.txt', 'rb') as f:
            data = f.read()
            encoding = chardet.detect(data)['encoding']
            sys_prompt = data.decode(encoding)

        assistant = "```python"
        if self.use_prompt:
            if self.use_my_sys_prompt:
                my_sys_prompt = self.my_sys_prompt
                sys_prompt += my_sys_prompt

            response = self.get_glm_response(prompt, do_sample=False, sys_prompt=sys_prompt, assistant=assistant)
        else:
            response = self.get_glm_response(prompt, do_sample=False, assistant=assistant)

        batch_size = min(self.batch_size, num_samples)

        # gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in response]
        gen_strs = [response.replace("\t", "    ")]
        return gen_strs

    def get_glm_response(self, prompt: str,
                          do_sample: bool = False,
                          sys_prompt: str = "You are an intelligent programming assistant to produce Python algorithmic solutions.",
                          assistant: str = ""):
        history = [{"role": "system", "content": sys_prompt},
                   {"role": "assistant", "content": assistant}]
        response, history = self.model.chat(self.tokenizer,
                                            prompt,
                                            history=history,
                                            max_length=1024,
                                            do_sample=do_sample)
        return response


# class QWEN2(DecoderBase):
#     def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
#         kwargs["direct_completion"] = True
#         super().__init__(name, **kwargs)
#         self.eos += ["\n```"]

#         self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
#         self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
#         if my_sys_prompt:
#             self.my_sys_prompt = my_sys_prompt

#     def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
#         print("********** do_sample: ", do_sample)
#         if do_sample:
#             assert self.temperature > 0, "Temperature must be greater than 0!"

#         with open('./prompt/system_prompt/qwen2_' + self.dataset.lower() + '.txt', 'r', encoding='utf-8') as f:
#             sys_prompt = f.read()

#         assistant = "<|im_start|>assistant\n```python"
#         if self.use_prompt:
#             if self.use_my_sys_prompt:
#                 my_sys_prompt = self.my_sys_prompt
#                 sys_prompt += my_sys_prompt
#             # prompt = f"Can you complete the following Python function?\n```python\n{prompt}\n```\n"

#             response = self.get_qwen_response(prompt, do_sample=False, sys_prompt=sys_prompt, add_generation_prompt=False, assistant=assistant)
#         else:
#             response = self.get_qwen_response(prompt, do_sample=False, add_generation_prompt=False, assistant=assistant)

#         # batch_size = min(self.batch_size, num_samples)
#         # gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in response]
#         gen_strs = [response.replace("\t", "    ")]
#         return gen_strs
class QWEN2(DecoderBase):
    def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
        kwargs["direct_completion"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        
        # ✅ 修复1：强制初始化
        self.my_sys_prompt = my_sys_prompt if my_sys_prompt is not None else ""
        self.use_my_sys_prompt = True if self.my_sys_prompt else False

    # ✅ 修复2：添加 sys_prompt_index 参数
    def codegen(self, sys_prompt_index: int, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        print("********** do_sample: ", do_sample)
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        # ✅ 修复3：根据 index 读取文件 (可选，如果您需要读取 0.txt)
        # 注意：需确保 get_txt_path 返回小写路径，或者文件夹本身是大写
        sys_prompt_path = './prompt/system_prompt/qwen2_' + self.dataset.lower() + '.txt'
        
        # 尝试读取对应 index 的 prompt 文件
        if sys_prompt_index is not None and sys_prompt_index >= 0:
             # 假设您已经按建议修改了 get_txt_path()
             indexed_path = self.get_txt_path() + str(sys_prompt_index) + ".txt"
             if os.path.exists(indexed_path):
                 print(f"Loading custom prompt from: {indexed_path}")
                 with open(indexed_path, 'r', encoding='utf-8') as f:
                     # 这里选择直接覆盖还是拼接，取决于您的需求
                     # sys_prompt = f.read().strip() 
                     self.my_sys_prompt = f.read().strip()
                     self.use_my_sys_prompt = True

        with open(sys_prompt_path, 'r', encoding='utf-8') as f:
            sys_prompt = f.read()

        assistant = "<|im_start|>assistant\n```python"
        if self.use_prompt:
            if self.use_my_sys_prompt:
                # 拼接逻辑
                sys_prompt += self.my_sys_prompt
            
            response = self.get_qwen_response(prompt, do_sample=False, sys_prompt=sys_prompt, add_generation_prompt=False, assistant=assistant)
        else:
            response = self.get_qwen_response(prompt, do_sample=False, add_generation_prompt=False, assistant=assistant)

        gen_strs = [response.replace("\t", "    ")]
        return gen_strs

    def get_qwen_response(self, prompt: str,
                          do_sample: bool = False,
                          sys_prompt: str = "You are an intelligent programming assistant to produce Python algorithmic solutions.",
                          assistant: str = "",
                          add_generation_prompt=True):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        if not add_generation_prompt:
            text = text + assistant
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=do_sample,
            top_p=0.95 if do_sample else 1.0,
            top_k=0,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class LLAMA3(DecoderBase):
    def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
        kwargs["direct_completion"] = True
        super().__init__(name, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if my_sys_prompt:
            self.my_sys_prompt = my_sys_prompt

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        print("********** do_sample: ", do_sample)
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        with open('./prompt/system_prompt/llama3_' + self.dataset.lower() + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()

        assistant = None
        # if self.dataset.lower() == 'humaneval':
        assistant = "```python"

        if self.use_prompt:
            # prompt = f"Can you complete the following Python function?\n```python\n{prompt}\n```\n"
            if self.use_my_sys_prompt:
                my_sys_prompt = self.my_sys_prompt
                sys_prompt += my_sys_prompt
            response = self.get_llama_response(prompt, do_sample=False, sys_prompt=sys_prompt, assistant=assistant)
        else:
            response = self.get_llama_response(prompt, do_sample=False, assistant=assistant)

        gen_strs = [response.replace("\t", "    ")]
        return gen_strs

    def get_llama_response(self, prompt: str,
                          do_sample: bool = False,
                          sys_prompt: str = "You are an intelligent programming assistant to produce Python algorithmic solutions",
                          assistant: str = None,
                          add_generation_prompt=True):
        if sys_prompt is not None:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        )

        if assistant is not None:
            assistant = self.tokenizer.encode(assistant, add_special_tokens=False)
            assistant = torch.LongTensor(assistant).unsqueeze(0)
            input_ids = torch.cat((input_ids, assistant), dim=-1)

        input_ids = input_ids.to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response


class GEMMA2(DecoderBase):
    def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if my_sys_prompt:
            self.my_sys_prompt = my_sys_prompt

    def codegen(self, prompt: str, do_sample: bool = False, num_samples: int = 200) -> List[str]:
        print("********** do_sample: ", do_sample)
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        with open('./prompt/system_prompt/gemma2_' + self.dataset.lower() + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()

        if self.use_prompt:
            # prompt = f"Can you complete the following Python function?\n```python\n{prompt}\n```\n"
            if self.use_my_sys_prompt:
                sys_prompt += self.my_sys_prompt
            response = self.get_gemma_response(prompt, do_sample=False, sys_prompt=sys_prompt)
        else:
            response = self.get_gemma_response(prompt, do_sample=False)

        gen_strs = [response.replace("\t", "    ")]
        return gen_strs

    def get_gemma_response(self,
                           prompt: str,
                           sys_prompt: str = "You are an intelligent programming assistant to produce Python algorithmic solutions.",
                           do_sample: bool = False
                           ):
        prompt = f"<bos><start_of_turn>system\n{sys_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```python"
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs.to(self.model.device),
            max_new_tokens=2048,
            do_sample=do_sample,
        )

        response = self.tokenizer.decode(outputs[0])
        response = self.gemma2_res_postprocess(response)
        return response

    def gemma2_res_postprocess(self, response):
        len_num = len('<start_of_turn>model')
        index = response.find('<start_of_turn>model')
        response = response[index + len_num + 1:]  # 1 represents '\n'

        index = response.find('<end_of_turn>')
        response = response[:index]
        return response


class Yi(DecoderBase):
    def __init__(self, name: str, my_sys_prompt: str = None, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if my_sys_prompt:
            self.my_sys_prompt = my_sys_prompt

    def codegen(self, prompt: str, do_sample: bool = False, num_samples: int = 200) -> List[str]:
        print("********** do_sample: ", do_sample)
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        with open('./prompt/system_prompt/yi1.5_' + self.dataset.lower() + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()

        if self.use_prompt:
            # prompt = f"Can you complete the following Python function?\n```python\n{prompt}\n```\n"
            if self.use_my_sys_prompt:
                my_sys_prompt = self.my_sys_prompt
                sys_prompt += my_sys_prompt
            response = self.get_yi_response(prompt, do_sample=False, sys_prompt=sys_prompt)
        else:
            response = self.get_yi_response(prompt, do_sample=False)

        gen_strs = [response.replace("\t", "    ")]
        return gen_strs

    def get_yi_response(self,
                        prompt: str,
                        sys_prompt: str = "You are an intelligent programming assistant to produce Python algorithmic solutions.",
                        do_sample: bool = False
                        ):
        prompt = f"{sys_prompt}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```python"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)

        output_ids = self.model.generate(
                                    input_ids.to(self.model.device),
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    max_new_tokens=2048,
                                    do_sample=do_sample,
                                    )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response


def make_model(
    model_type: str,
    model_size: str,
    model_path: str,
    batch_size: int = 1,
    temperature: float = 0.8,
    my_sys_prompt: str = None,
    dataset: str = None,
):
    if model_type == "glm3":
        return GLM(
                    batch_size=batch_size,
                    name=model_path,
                    temperature=temperature,
                    max_new_tokens=2048,
                    my_sys_prompt=my_sys_prompt,
                    dataset=dataset,
        )
    elif model_type == "qwen2" or model_type == "qwen3":
        return QWEN2(
            batch_size=batch_size,
            name=model_path,
            temperature=temperature,
            max_new_tokens=2048,
            my_sys_prompt=my_sys_prompt,
            dataset=dataset,
        )
    elif model_type == "llama3":
        return LLAMA3(
            batch_size=batch_size,
            name=model_path,
            temperature=temperature,
            max_new_tokens=2048,
            my_sys_prompt=my_sys_prompt,
            dataset=dataset,
        )
    elif model_type == "gemma2":
        return GEMMA2(
            batch_size=batch_size,
            name=model_path,
            temperature=temperature,
            max_new_tokens=2048,
            my_sys_prompt=my_sys_prompt,
            dataset=dataset,
        )
    elif model_type == "yi1.5":
        return Yi(
            batch_size=batch_size,
            name=model_path,
            temperature=temperature,
            max_new_tokens=2048,
            my_sys_prompt=my_sys_prompt,
            dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid model name: {model_type}@{model_size}")


def merge_list_wo_rep(ori_str, add_str, split_token=', '):
    target_str = ''
    ori_list = ori_str.split(split_token)
    add_list = add_str.split(split_token)
    for token in add_list:
        if token not in ori_list:
            ori_list.append(token)
    for token in ori_list:
        target_str += token + split_token
    return target_str


if __name__ == "__main__":
    gemma_humaneval_prompt_new_1 = "1, (, [,     , i, "
    gemma_humaneval_prompt_new_2 = "(, ], _,  , ):, "
    after_str = merge_list_wo_rep(gemma_humaneval_prompt_new_1, gemma_humaneval_prompt_new_2)
    print(after_str)
