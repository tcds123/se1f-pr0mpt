from abc import ABC, abstractmethod
from transformers import pipeline


class BaseGeneration(ABC):
    def __init__(self,
                 model,
                 tokenizer,
                 device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        pass


class GLM3Generation(BaseGeneration):
    def __init__(self, *args):
        super().__init__(*args)

    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        history = [{"role": "system", "content": sys_prompt}]
        response, history = self.model.chat(self.tokenizer,
                                            prompt,
                                            history=history,
                                            max_length=65536,
                                            do_sample=do_sample)
        return response


class QWEN2Generation(BaseGeneration):
    def __init__(self, *args):
        super().__init__(*args)

    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=do_sample
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


class LLAMA3Generation(BaseGeneration):
    def __init__(self, *args):
        super().__init__(*args)

    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=do_sample,
        )
        response = outputs[0][input_ids.shape[-1]:]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        return response


class GEMMA2Generation(BaseGeneration):
    def __init__(self, *args):
        super().__init__(*args)

    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        prompt = f"<bos><start_of_turn>system\n{sys_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

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


class YiGeneration(BaseGeneration):
    def __init__(self, *args):
        super().__init__(*args)

    def generate(self, prompt: str, sys_prompt="You are a helpful assistant.", do_sample=False):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device),
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    max_new_tokens=2048,
                                    )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response
