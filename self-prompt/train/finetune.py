#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Adapted from 


import logging
import os
import random
import sys
import json
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer import PrefixTrainer

from arguments import ModelArguments, DataTrainingArguments

from preprocess_utils import InputOutputDataset, GLMDataset, Qwen2Dataset, Llama3Dataset, Gemma2Dataset, YiDataset

# from datasets import load_metric
import numpy as np

logger = logging.getLogger(__name__)
sys.path.append('./my_model')


TRAIN_EVAL_PER = [0.6, 0.2, 0.2]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
    config.step = training_args.gradient_accumulation_steps

    # TODO
    config.sp_token_num = 0

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    prompt_template = None
    if 'glm' in model_args.model_name:
        with open('./prompt/system_prompt/' + model_args.model_name + '_' + data_args.dataset_name + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
    elif 'qwen' in model_args.model_name:
        with open('./prompt/system_prompt/' + model_args.model_name + '_' + data_args.dataset_name + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
        # qwen2_user_prompt = {"prompt": "Can you complete the following Python function?\n```python\n{prompt}\n```\n"}
        qwen2_user_prompt = {"prompt": "{prompt}"}
        assistant = "```python"

        qwen2_prompt_template = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": qwen2_user_prompt},
            {"role": "assistant", "content": assistant}
        ]
        prompt_template = qwen2_prompt_template
    elif 'llama' in model_args.model_name:
        with open('./prompt/system_prompt/' + model_args.model_name + '_' + data_args.dataset_name + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
    elif 'gemma' in model_args.model_name:
        with open('./prompt/system_prompt/' + model_args.model_name + '_' + data_args.dataset_name + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
        gemma2_prompt_template = "<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n```python"
        prompt_template = gemma2_prompt_template
    elif 'yi' in model_args.model_name:
        with open('./prompt/system_prompt/' + model_args.model_name + '_' + data_args.dataset_name + '.txt', 'r', encoding='utf-8') as f:
            sys_prompt = f.read()
        yi_prompt_template = "{system_prompt}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```python"
        prompt_template = yi_prompt_template
    else:
        raise ValueError('Currently glm, qwen, llama, gemma, yi model supported...')

    extra_len = model_args.extra_len

    sys_prompt_len = calc_tokenize_len(tokenizer, sys_prompt)
    config.sys_prompt_is_list = False
    if isinstance(sys_prompt_len, list):
        assert len(sys_prompt_len) == 2
        config.sys_prompt_is_list = True
    config.sys_prompt_len = sys_prompt_len
    config.extra_len = extra_len
    config.train_mode = True
    config.dataset_name = data_args.dataset_name
    model_name = model_args.model_name

    # model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    with open(data_args.train_file, "r", encoding="utf-8") as f:
        if data_args.train_file.endswith(".json"):
            data = json.load(f)
        elif data_args.train_file.endswith(".jsonl"):
            data = [json.loads(line) for line in f]

    # TODO shuffle
    if data_args.dataset_name == "humaneval":
        train_num = int(len(data) * TRAIN_EVAL_PER[0])
        val_num = int(len(data) * TRAIN_EVAL_PER[1])
        test_num = len(data) - train_num - val_num
        train_data = data[:train_num+val_num]

        val_index = dict()
        test_index = dict()
        val_index["index"] = []
        test_index["index"] = []
        for i in range(train_num, train_num+val_num):
            val_index["index"].append("HumanEval/" + str(i))
        for i in range(train_num + val_num, len(data)):
            test_index["index"].append("HumanEval/" + str(i))

        val_path = data_args.train_file.split('train.jsonl')[0] + 'val.jsonl'
        test_path = data_args.train_file.split('train.jsonl')[0] + 'test.jsonl'

        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_index, f)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_index, f)
    elif data_args.dataset_name == "mbpp":
        train_data = data
        train_num = [601, 975]
        val_num = [511, 601]
        test_num = [11, 511]

        val_index = dict()
        test_index = dict()
        val_index["index"] = []
        test_index["index"] = []
        for i in range(val_num[0], val_num[1]):
            val_index["index"].append("Mbpp/" + str(i))
        for i in range(test_num[0], test_num[1]):
            test_index["index"].append("Mbpp/" + str(i))

        val_path = data_args.train_file.split('train.jsonl')[0] + 'val.jsonl'
        test_path = data_args.train_file.split('train.jsonl')[0] + 'test.jsonl'

        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_index, f)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_index, f)
    config.val_path = val_path
    config.test_path = test_path

    use_half = False
    if use_half:
        len_train_data = len(train_data)
        len_train_data = int(len_train_data / 2)

        train_data = random.sample(train_data, len_train_data)

    sample_ids_list = sample_all_token(tokenizer, train_data, 'response')
    config.sample_ids_list = sample_ids_list
    config.use_all = False

    if model_name == 'glm3':
        sp_token_num = (5, 0)
        config.sp_token_num = sp_token_num
        # config.auto_map = {
        #     "AutoModelForCausalLM": "glm3_model.MyChatGLMForConditionalGeneration",
        # }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            trust_remote_code=True
        )
        model = model.float()
        train_dataset = GLMDataset(
            data=train_data,
            dataset_name=data_args.dataset_name,
            device=model.device,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            sys_prompt=sys_prompt,
            sys_prompt_len=sys_prompt_len,
            extra_len=extra_len,
            prompt_template=prompt_template,
            add_generation_prompt=False,
            sp_token_num=sp_token_num,
        )
    elif model_name == 'qwen2':
        sp_token_num = (2, 1)
        config.sp_token_num = sp_token_num
        config.auto_map = {
            "AutoModelForCausalLM": "qwen2_model.MyQwen2ForCausalLM",
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype="auto",
            trust_remote_code=True
        )
        train_dataset = Qwen2Dataset(
            data=train_data,
            dataset_name=data_args.dataset_name,
            device=model.device,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            sys_prompt=sys_prompt,
            sys_prompt_len=sys_prompt_len,
            extra_len=extra_len,
            prompt_template=prompt_template,
            add_generation_prompt=False,
            sp_token_num=sp_token_num,
        )
    elif model_name == 'llama3':
        sp_token_num = (5, 1)
        config.sp_token_num = sp_token_num
        config.sp_token = [128000, 128001, 128006, 128007, 128009]
        config.auto_map = {
            "AutoModelForCausalLM": "llama3_model.MyLlamaForCausalLM",
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype="auto",
            trust_remote_code=True
        )
        train_dataset = Llama3Dataset(
            data=train_data,
            dataset_name=data_args.dataset_name,
            device=model.device,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            sys_prompt=sys_prompt,
            sys_prompt_len=sys_prompt_len,
            extra_len=extra_len,
            prompt_template=prompt_template,
            add_generation_prompt=True,
            sp_token_num=sp_token_num,
        )
    elif model_name == 'gemma2':
        sp_token_num = (4, 1)
        config.sp_token_num = sp_token_num
        config.auto_map = {
            "AutoModelForCausalLM": "gemma2_model.MyGemma2ForCausalLM",
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype="auto",
            trust_remote_code=True
        )
        train_dataset = Gemma2Dataset(
            data=train_data,
            dataset_name=data_args.dataset_name,
            device=model.device,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            sys_prompt=sys_prompt,
            sys_prompt_len=sys_prompt_len,
            extra_len=extra_len,
            prompt_template=prompt_template,
            add_generation_prompt=True,
            sp_token_num=sp_token_num,
        )
    elif model_name == 'yi1.5':
        sp_token_num = (0, 0)
        config.sp_token_num = sp_token_num
        config.auto_map = {
            "AutoModelForCausalLM": "yi_model.MyYiForCausalLM",
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype="auto",
            trust_remote_code=True
        )
        train_dataset = YiDataset(
            data=train_data,
            dataset_name=data_args.dataset_name,
            device=model.device,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            sys_prompt=sys_prompt,
            sys_prompt_len=sys_prompt_len,
            extra_len=extra_len,
            prompt_template=prompt_template,
            add_generation_prompt=True,
            sp_token_num=sp_token_num,
        )
    else:
        raise ValueError('Only qwen2, llama3, gemma2, yi1.5 model supported...')



    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    # Initialize our Trainer
    trainer = PrefixTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        save_changed=True,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()


def calc_tokenize_len(tokenizer, input_str):
    if isinstance(input_str, str):
        ids = tokenizer.encode(text=input_str, add_special_tokens=False)
        # print(ids)
        # print(tokenizer.decode(ids))
        ids_len = len(ids)
        # print(ids_len)
        return ids_len
    elif isinstance(input_str, list):
        input_str_list = input_str
        ids_len_list = []
        for input_str in input_str_list:
            ids = tokenizer.encode(text=input_str, add_special_tokens=False)
            ids_len = len(ids)
            ids_len_list.append(ids_len)
        return ids_len_list


def _calc_tokenize_len(tokenizer, input_str):
    ids = tokenizer.encode(text=input_str, add_special_tokens=False)
    ids_len = len(ids)
    return ids_len


def sample_all_token(tokenizer, train_data, name="prompt"):
    prompt_ids_list = []
    for data in train_data:
        prompt = data[name]
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids_list += ids

    return prompt_ids_list




if __name__ == "__main__":
    main()
