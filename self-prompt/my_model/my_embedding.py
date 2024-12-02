import os.path

from transformers import AutoTokenizer
from transformers.utils import logging
import torch
from torch import nn
import copy
import random
import json
import sys

logger = logging.get_logger(__name__)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_MAPPING = {
    #  Can be either repo's name or /path/to/model
    "glm3": {
        "chat": "/data/team/zongwx1/llm_models/chatglm3-6b"
    },
    "qwen2": {
        "chat": "/data/team/zongwx1/llm_models/qwen2-7b-instruct"
    },
    "llama3": {
        "chat": "/data/team/zongwx1/llm_models/llama3-8b-instruct"
    },
    "gemma2": {
        "chat": "/data/team/zongwx1/llm_models/gemma-2-9b-it"
    },
    "yi1.5": {
        "chat": "/data/team/zongwx1/llm_models/Yi-1.5-9B-Chat"
    },
}


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config, emb, tokenizer, model_name):
        super(Embedding, self).__init__()

        self.my_word_embeddings = copy.deepcopy(emb)
        self.hidden_size = config.hidden_size
        self.padded_vocab_size = config.vocab_size
        self.torch_dtype = config.torch_dtype
        self.tokenizer = tokenizer
        self.model_name = model_name

        self.is_train_sys_prompt_mode = False
        if hasattr(config, 'step'):
            self.is_train_sys_prompt_mode = True
            self.device = 'cuda'
            self.w_emb = None
            self.sys_prompt_len = config.sys_prompt_len
            self.sys_prompt_is_list = config.sys_prompt_is_list
            if hasattr(config, 'extra_len'):
                self.extra_len = config.extra_len
            else:
                self.extra_len = 50
            self.extra_ids = None
            self.base_skip = config.step
            self.skip = config.step
            self.sp_token_num = config.sp_token_num
            if not self.sys_prompt_is_list:
                self.skip_num_front = self.sp_token_num[0] + self.sys_prompt_len + self.extra_len
                self.skip_num_back = self.sp_token_num[1]
            self.label_set = set()
            self.new_label_set = set()
            self.ignore_ids = -100
            self.should_careful_list = []
            self.record_epoch = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            self.ignore_step = config.step
            if hasattr(config, 'sp_token'):
                self.sp_token = config.sp_token
            self.should_careful_dict = {}
            self.dataset_name = config.dataset_name
            self.pre_loss = None
            self.weight_dict = {}
            self.skip_count = 1
            sample_ids_list = config.sample_ids_list
            self.sample_ids = torch.LongTensor(sample_ids_list).unsqueeze(0)
            self.use_all = config.use_all
            self.val_path = config.val_path
            self.test_path = config.test_path
            self.best_results = None
            self.best_extra_prompt = None
            self.run_epoch = [i for i in range(20)]
            self.model_runner = None

    @property
    def weight(self):
        return self.my_word_embeddings.weight

    def update(self, weight):
        self.my_word_embeddings.weight = weight

    def forward(self, input_ids):
        if self.w_emb is None:
            self.w_emb = nn.Embedding(
                self.padded_vocab_size,
                self.hidden_size,
                dtype=self.torch_dtype,
                device=self.device,
                _weight=copy.deepcopy(self.my_word_embeddings.weight.detach())
            )
        if self.extra_len > 0 and self.extra_ids is None:
            self.extra_ids = self.initialize_extra_ids(input_ids, self.extra_len).to(self.device)

        embeddings = self.my_train_sys_emb(input_ids)

        return embeddings

    def my_train_sys_emb(self, input_ids):
        w_embeddings = self.w_emb
        if not self.sys_prompt_is_list:
            input_embedding = w_embeddings(input_ids[:, self.skip_num_front + self.skip_num_back:])
            sys_tensor = input_ids[:, :self.sp_token_num[0] + self.sys_prompt_len]

            if self.extra_len > 0:
                sys_tensor = torch.cat((sys_tensor, self.extra_ids, input_ids[:, self.skip_num_front:self.skip_num_front+self.skip_num_back]), dim=1)
            else:
                sys_tensor = torch.cat((sys_tensor, input_ids[:, self.skip_num_front:self.skip_num_front + self.skip_num_back]),dim=1)
            sys_emb = self.my_word_embeddings(sys_tensor)
        else:
            prompt_len = torch.load('./pt_file/prompt_len.pt')
            len_input = prompt_len - self.sp_token_num - self.sys_prompt_len[0] - self.sys_prompt_len[1] - self.extra_len
            input_embedding_front = w_embeddings(input_ids[:, self.sp_token_num + self.sys_prompt_len[0]:self.sp_token_num + self.sys_prompt_len[0] + len_input])
            input_embedding_back = w_embeddings(input_ids[:, prompt_len:])

            sys_tensor_front = input_ids[:, :self.sp_token_num + self.sys_prompt_len[0]]
            sys_tensor_back = input_ids[:, self.sp_token_num + self.sys_prompt_len[0] + len_input:self.sp_token_num + self.sys_prompt_len[0] + len_input + self.sys_prompt_len[1]]
            sys_tensor_back = torch.cat((sys_tensor_back, self.extra_ids), dim=1)

            sys_emb_front = self.my_word_embeddings(sys_tensor_front)
            sys_emb_back = self.my_word_embeddings(sys_tensor_back)

            sys_emb = [sys_emb_front, sys_emb_back]

        if self.skip_count < 1:
            label = self.get_shift_labels()
            self.update_new_label_set(label)
        self.skip_count -= 1
        if self.ignore_step <= 0:
            self.get_change_most_input_ids(w_embeddings.weight.detach(), self.my_word_embeddings.weight.detach())
        self.ignore_step -= 1

        if self.skip <= 1:
            self.print_input_ids(sys_emb, w_emb=w_embeddings)
            self.skip = self.base_skip + 1
        self.skip -= 1

        if not self.sys_prompt_is_list:
            embeddings = torch.cat((sys_emb, input_embedding), dim=1)
        else:
            embeddings = torch.cat((sys_emb_front, input_embedding_front, sys_emb_back, input_embedding_back), dim=1)

        # embeddings = embeddings.transpose(0, 1)

        return embeddings

    def update_new_label_set(self, label):
        if label is not None:
            label = label[0].tolist()
            self.new_label_set = self.label_set | set(label)

    def emb_2_input_ids(self, embeddings, w_emb, add_special_tokens=True):
        w_tran = w_emb.weight.detach().float().transpose(0, 1)
        input_ids_logits = torch.matmul(embeddings.float(), w_tran)

        if not add_special_tokens:
            special_tokens_len = len(self.tokenizer.get_prefix_tokens())
            return input_ids_logits[:, special_tokens_len:]

        return input_ids_logits

    def get_change_most_input_ids(self, ori_emb_table, cur_emb_table):
        # print("torch.topk(ori_emb_table[0], 10): ", torch.topk(ori_emb_table[:3], 10))
        # print("torch.topk(cur_emb_table[0], 10): ", torch.topk(cur_emb_table[:3], 10))
        if self.get_epoch() < 1:
            if not torch.all(torch.eq(ori_emb_table, cur_emb_table)):
                print("不完全一样")
            else:
                print("完全一样")
                return

        if self.skip <= 1:
            self.label_set = self.new_label_set
            self.new_label_set = set()
            self.get_change_most_ids(self.label_set, ori_emb_table, cur_emb_table)

    def get_change_most_ids(self,
                            input_ids,
                            ori_emb_table,
                            cur_emb_table,
                            topk=5,
                            collect_per_step=False,
                            check_loss_dir=True,
                            skip_first_epoch=True,
                            apply_weight=True,
                            dynamic_expand=False,
                            use_all_vocab=True,):
        epoch = self.get_epoch()
        if dynamic_expand:
            topk = epoch
        diff_emb_table = cur_emb_table - ori_emb_table
        sum_diff_emb_table = torch.sum(abs(diff_emb_table), dim=1).tolist()
        print(sorted(sum_diff_emb_table, reverse=True)[:10])
        new_sum_diff_emb_dict = {}
        if self.use_all:
            sample_list = list(self.sample_ids[0])
        elif use_all_vocab:
            sample_list = [i for i in range(len(sum_diff_emb_table))]
        else:
            sample_list = input_ids

        print(f"sample_list len = {len(sample_list)}")
        for ids in sample_list:
            if ids in self.weight_dict:
                new_sum_diff_emb_dict[ids] = self.weight_dict[ids] * sum_diff_emb_table[ids]
            else:
                new_sum_diff_emb_dict[ids] = sum_diff_emb_table[ids]
        if check_loss_dir:
            cur_loss = torch.load('./pt_file/epoch_loss.pt')
            if self.pre_loss is None:
                self.pre_loss = cur_loss
            else:
                if apply_weight:
                    weight = 1 - ((cur_loss - self.pre_loss) / max(cur_loss, self.pre_loss))
                    for ids in input_ids:
                        if ids in self.weight_dict:
                            self.weight_dict[ids] = self.weight_dict[ids] * weight
                        else:
                            self.weight_dict[ids] = weight
                    print("weight: ", round(weight, 4))
                    # print("weight_dict: ", self.weight_dict)
                if cur_loss > self.pre_loss:
                    print("***************** 检测到loss上升 *****************")
                    self.pre_loss = cur_loss
                    return
                else:
                    self.pre_loss = cur_loss

        if skip_first_epoch:
            if epoch < 1:
                print("***************** epoch < 1, skip *****************")
                return

        sorted_dict = sorted(new_sum_diff_emb_dict.items(), key=lambda d: d[1], reverse=True)
        max_values = sorted_dict[:topk]
        max_indices = []
        for dict in max_values:
            max_indices.append(dict[0])

        for max_index in max_indices:
            print([max_index])
            print([self.tokenizer.decode([max_index])])
            print("----------")

            if max_index not in self.should_careful_list:
                self.should_careful_list.append(max_index)

        if collect_per_step:
            self.should_careful_list = max_indices
        should_careful_structure = self.should_careful_list

        print(should_careful_structure)
        print("*********** self.should_careful_structure: ", end='')
        extra_prompt = ""
        for id in should_careful_structure:
            print(self.tokenizer.decode([id]), end=', ')
            extra_prompt += self.tokenizer.decode([id]) + ', '

        if epoch > self.run_epoch[0]:
            self.evaluation(extra_prompt)
            self.run_epoch = self.run_epoch[1:]
        if epoch >= 20:
            self.test()

        # if epoch >= self.record_epoch[0]:
        #     if not os.path.exists('./txt/' + self.model_name):
        #         os.mkdir('./txt/' + self.model_name)
        #     if not os.path.exists('./txt/' + self.model_name + '/' + self.dataset_name):
        #         os.mkdir('./txt/' + self.model_name + '/' + self.dataset_name)
        #     with open('./txt/' + self.model_name + '/' + self.dataset_name + '/' + str(self.record_epoch[0])+'.txt', 'w', encoding='utf-8') as f:
        #         for id in should_careful_structure:
        #             print(self.tokenizer.decode([id]), end=', ', file=f)
        #     self.record_epoch = self.record_epoch[1:]
        #     f.close()

        print("\n")

    def print_2_dict(self, epoch, max_indices):
        if epoch not in self.should_careful_dict:
            self.should_careful_dict[epoch] = max_indices
        else:
            for max_index in max_indices:
                if max_index not in self.should_careful_dict[epoch]:
                    self.should_careful_dict[epoch].append()
            self.should_careful_dict[epoch] += max_indices

    def get_epoch(self):
        epoch = torch.load('./pt_file/epoch.pt')
        return epoch

    def get_shift_labels(self):
        shift_labels = torch.load('./pt_file/shift_labels.pt')
        torch.save(None, './pt_file/shift_labels.pt')
        if shift_labels is not None:
            clean_shift_labels_list = []
            shift_labels_list = shift_labels[0].tolist()
            for shift_labels in shift_labels_list:
                if shift_labels != self.ignore_ids:
                    clean_shift_labels_list.append(shift_labels)
            shift_labels = torch.LongTensor([clean_shift_labels_list])
            # print("shift_labels: ", shift_labels)
        return shift_labels

    def save_pt(self, value, name):
        torch.save(value, './pt_file/' + name + '.pt')

    def print_input_ids(self, emb, w_emb):
        if not self.sys_prompt_is_list:
            self._print_input_ids(emb, w_emb)
        else:
            emb_list = emb
            for emb in emb_list:
                self._print_input_ids(emb, w_emb)

    def _print_input_ids(self, emb, w_emb):
        temp_input_ids_logits = self.emb_2_input_ids(emb, w_emb).detach()
        sys_input_ids = temp_input_ids_logits.float()
        sys_input_ids = torch.argmax(sys_input_ids, dim=2)
        sys_input_ids_list = []
        for sys_input_id in sys_input_ids[0]:
            decoded_sys_input_id = self.tokenizer.decode([sys_input_id])
            sys_input_ids_list.append(decoded_sys_input_id)
        print("*************** sys_input_ids: ", sys_input_ids[0])
        print("*************** decoded_sys_input_ids: ", sys_input_ids_list)

    def initialize_extra_ids(self, input_ids, extra_len: int, random_sample=False):
        if random_sample:
            extra_ids = self.initialize_random(extra_len)
            return extra_ids
        if self.use_all:
            extra_ids = self.initialize_use_all(self.sample_ids, extra_len)
            return extra_ids
        if not self.sys_prompt_is_list:
            should_skip_len = self.sp_token_num[0] + self.sp_token_num[1] + self.sys_prompt_len + self.extra_len
            input_ids = input_ids[:, should_skip_len:]
        else:
            prompt_len = torch.load('./pt_file/prompt_len.pt')
            len_input = prompt_len - self.sp_token_num - self.sys_prompt_len[0] - self.sys_prompt_len[1] - self.extra_len
            input_ids = input_ids[:, self.sp_token_num + self.sys_prompt_len[0]:self.sp_token_num + self.sys_prompt_len[0]+len_input]

        clean_input_ids = self.kill_meaningless_ids(input_ids)
        sample_target = list(clean_input_ids[0])
        if len(sample_target) < extra_len:
            while len(sample_target) < extra_len:
                copy_id = random.choice(sample_target)
                sample_target.append(copy_id)

        sample_ids = random.sample(sample_target, extra_len)
        sample_ids_list = []
        for sample_id in sample_ids:
            decoded_sample_id = self.tokenizer.decode(sample_id.tolist())
            sample_ids_list.append(decoded_sample_id)
        print("**************** extra_ids: ", sample_ids_list)
        extra_ids = torch.LongTensor(sample_ids).unsqueeze(0)
        with open('./txt/' + self.model_name + '/' + self.dataset_name + '/extra_ids.txt', 'w',
                  encoding='utf-8') as f:
            print(extra_ids[0].tolist(), file=f)
            for id in extra_ids[0].tolist():
                print([self.tokenizer.decode([id])], end=', ', file=f)
        return extra_ids

    def initialize_random(self, extra_len):
        input_ids = torch.LongTensor([[i for i in range(self.padded_vocab_size)]])
        clean_input_ids = self.kill_meaningless_ids(input_ids)
        sample_target = list(clean_input_ids[0])
        if len(sample_target) < extra_len:
            while len(sample_target) < extra_len:
                copy_id = random.choice(sample_target)
                sample_target.append(copy_id)

        sample_ids = random.sample(sample_target, extra_len)
        sample_ids_list = []
        for sample_id in sample_ids:
            decoded_sample_id = self.tokenizer.decode(sample_id.tolist())
            sample_ids_list.append(decoded_sample_id)
        print("**************** extra_ids: ", sample_ids_list)
        extra_ids = torch.LongTensor(sample_ids).unsqueeze(0)
        with open('./txt/' + self.model_name + '/' + self.dataset_name + '/extra_ids.txt', 'w',
                  encoding='utf-8') as f:
            print(extra_ids[0].tolist(), file=f)
            for id in extra_ids[0].tolist():
                print([self.tokenizer.decode([id])], end=', ', file=f)
        return extra_ids

    def initialize_use_all(self, input_ids, extra_len: int):
        clean_input_ids = self.kill_meaningless_ids(input_ids)
        sample_target = list(clean_input_ids[0])
        if len(sample_target) < extra_len:
            while len(sample_target) < extra_len:
                copy_id = random.choice(sample_target)
                sample_target.append(copy_id)

        sample_ids = random.sample(sample_target, extra_len)
        sample_ids_list = []
        for sample_id in sample_ids:
            decoded_sample_id = self.tokenizer.decode(sample_id.tolist())
            sample_ids_list.append(decoded_sample_id)
        print("**************** extra_ids: ", sample_ids_list)
        extra_ids = torch.LongTensor(sample_ids).unsqueeze(0)
        with open('./txt/' + self.model_name + '/' + self.dataset_name + '/extra_ids.txt', 'w',
                  encoding='utf-8') as f:
            print(extra_ids[0].tolist(), file=f)
            for id in extra_ids[0].tolist():
                print([self.tokenizer.decode([id])], end=', ', file=f)
        return extra_ids

    def kill_meaningless_ids(self, input_ids):
        input = input_ids[0]
        input = list(set(input.tolist()))

        kill_sp_token = False
        if kill_sp_token:
            new_input = []
            for id in input:
                if id in self.sp_token:
                    continue
                new_input.append(id)
            input = new_input

        input_ids = torch.LongTensor(input).unsqueeze(0)

        return input_ids

    def evaluation(self, extra_prompt):
        save_path = './outputs/val/' + self.dataset_name + '/' + self.model_name + '/'
        os.makedirs(save_path, exist_ok=True)

        index_path = self.val_path
        with open(index_path, "r") as f:
            for line in f:
                index = json.loads(line)["index"]

        # step 1: code generate
        self.code_generate(save_path, extra_prompt, index)

        # step 2: evaluation
        cur_results = self.code_evaluate(dataset=self.dataset_name, samples=save_path + self.model_name + '.jsonl')
        cur_score = cur_results["pass_at_k"]["base"]["pass@1"]
        print(f"********* cur_extra_prompt: {extra_prompt} *********")
        print(f"********* cur_results: {cur_score} *********")
        if not self.best_results:
            self.best_results = cur_score
            self.best_extra_prompt = extra_prompt
        else:
            if cur_score > self.best_results:
                self.best_results = cur_score
                self.best_extra_prompt = extra_prompt
        print(f"********* best_extra_prompt: {self.best_extra_prompt} *********")
        print(f"********* best_results: {self.best_results} *********")

    def test(self):
        save_path = './outputs/test/' + self.dataset_name + '/' + self.model_name + '/'
        os.makedirs(save_path, exist_ok=True)

        index_path = self.test_path
        with open(index_path, "r") as f:
            for line in f:
                index = json.loads(line)["index"]

        # step 1: code generate
        self.code_generate(save_path, self.best_extra_prompt, index)

        # step 2: evaluation
        cur_results = self.code_evaluate(dataset=self.dataset_name, samples=save_path + self.model_name + '.jsonl')
        cur_score = cur_results["pass_at_k"]["base"]["pass@1"]

        print(f"********* extra_prompt: {self.best_extra_prompt} *********")
        print(f"********* test_score: {cur_score} *********")
        exit(0)

    def code_generate(self, target_path, extra_prompt, index):
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_plus.model import make_model

        n_samples = 1
        batch_size = 1
        temperature = 0
        greedy = True

        if "glm" in self.model_name.lower():
            model_type = "glm3"
        elif "qwen" in self.model_name.lower():
            model_type = "qwen2"
        elif "llama" in self.model_name.lower():
            model_type = "llama3"
        elif "gemma" in self.model_name.lower():
            model_type = "gemma2"
        elif "yi" in self.model_name.lower():
            model_type = "yi1.5"

        model_size = "chat"
        model_path = MODEL_MAPPING[model_type][model_size]

        if not self.model_runner:
            # Model creation
            self.model_runner = make_model(
                model_type=model_type,
                model_size=model_size,
                model_path=model_path,
                dataset=self.dataset_name,
                batch_size=batch_size,
                temperature=temperature,
                my_sys_prompt=extra_prompt,
            )
            device = torch.device("cuda:1")
            self.model_runner.model.to(device)

        with torch.no_grad():
            self.codegen(
                target_path=target_path,
                dataset=self.dataset_name,
                index=index,
                greedy=greedy,
                model=self.model_runner,
                n_samples=n_samples,
            )

        return target_path

    def codegen(self,
                target_path: str,
                model,
                dataset: str,
                index,
                greedy=False,
                n_samples=1,
                id_range=None,
                version="default",
                resume=False,):
        from evalplus.data import (
            get_human_eval_plus,
            get_mbpp_plus,
            get_evalperf_data,
        )
        from evalplus.sanitize import sanitize
        from evalplus.utils import progress

        jsonl_fmt = True

        all_tasks_complete = False
        if jsonl_fmt:
            target_path += self.model_name + ".jsonl"
        #     if os.path.isfile(target_path):
        #         task_counts = {}
        #         with open(target_path, "r") as f:
        #             for line in f:
        #                 if not line.strip():
        #                     continue
        #                 data = json.loads(line)
        #                 task_id = data["task_id"]
        #                 task_counts[task_id] = task_counts.get(task_id, 0) + 1
        #
        #             all_tasks_complete = all(
        #                 task_counts.get(task_id, 0) >= n_samples
        #                 for task_id in dataset_dict.keys()
        #             )
        #
        # if all_tasks_complete:
        #     print("All samples are already cached. Skipping codegen.")
        #     return target_path

        task2nexist = {}
        if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
            with open(target_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    task_id = json.loads(line)["task_id"]
                    task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

        if target_path.endswith(".jsonl"):
            raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        else:
            raw_target_path = target_path + ".raw"
            os.makedirs(target_path, exist_ok=True)

        print(f"Sanitized code outputs will be saved to {target_path}")
        print(f"Raw outputs will be saved to {raw_target_path}")

        with progress(dataset) as p:
            if dataset == "humaneval":
                dataset = get_human_eval_plus(version=version)
            elif dataset == "mbpp":
                dataset = get_mbpp_plus(version=version)
            elif dataset == "evalperf":
                original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
                dataset = {k: original_dataset[k] for k in get_evalperf_data()}
                assert id_range is None, "id_range not supported for evalperf"
            else:
                raise ValueError(f"Invalid dataset {dataset}")

            my_dataset = dict()
            for i in index:
                my_dataset[i] = dataset[i]
            dataset = my_dataset

            for task_id, task in p.track(dataset.items()):
                if id_range is not None:
                    id_num = int(task_id.split("/")[1])
                    low, high = id_range
                    if id_num < low or id_num >= high:
                        p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                        continue

                if not target_path.endswith(".jsonl"):
                    p_name = task_id.replace("/", "_")
                    os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                    task2nexist[task_id] = len(
                        [
                            f
                            for f in os.listdir(os.path.join(target_path, p_name))
                            if f.endswith(".py")
                        ]
                    )

                n_more_samples = n_samples
                log = f"Codegen: {task_id} @ {model}"
                if resume and task2nexist.get(task_id, 0) > 0:
                    log += f" (resuming from {task2nexist[task_id]})"
                    n_more_samples -= task2nexist[task_id]

                p.console.print(log)

                sidx = n_samples - n_more_samples
                while sidx < n_samples:
                    prompt = task["prompt"].strip() + "\n"
                    outputs = model.codegen(
                        prompt,
                        do_sample=not greedy,
                        num_samples=n_samples - sidx,
                    )
                    assert outputs, "No outputs from model!"
                    for impl in outputs:
                        solution = prompt + impl if model.is_direct_completion() else impl
                        sanitized_solution = sanitize(
                            solution, entrypoint=task["entry_point"]
                        )
                        if target_path.endswith(".jsonl"):
                            # Writing the sanitized version
                            with open(target_path, "a") as f:
                                f.write(
                                    json.dumps(
                                        {"task_id": task_id, "solution": sanitized_solution}
                                    )
                                    + "\n"
                                )

                            # Writing the raw version
                            with open(raw_target_path, "a") as f:
                                f.write(
                                    json.dumps({"task_id": task_id, "solution": solution})
                                    + "\n"
                                )
                        else:
                            # Writing the sanitized version
                            with open(
                                    os.path.join(target_path, p_name, f"{sidx}.py"),
                                    "w",
                                    encoding="utf-8",
                            ) as f:
                                f.write(sanitized_solution)

                            # Writing the raw version
                            with open(
                                    os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                                    "w",
                                    encoding="utf-8",
                            ) as f:
                                f.write(solution)
                        sidx += 1

    def code_evaluate(
        self,
        dataset: str,
        samples = None,
        base_only: bool = False,
        parallel = None,
        i_just_wanna_run: bool = False,
        test_details: bool = False,
        min_time_limit: float = 1.0,
        gt_time_limit_factor: float = 4.0,
        mini: bool = False,
        noextreme: bool = False,
        version: str = "default",
        **model_kwargs,
    ):
        import json
        import multiprocessing
        import os
        import threading
        import time
        from collections import Counter, defaultdict
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from datetime import datetime
        from typing import Any, Dict, List, Optional, Tuple
        from warnings import warn

        import numpy as np
        from termcolor import cprint
        from tqdm import tqdm

        from evalplus.data import (
            get_human_eval_plus,
            get_human_eval_plus_hash,
            get_mbpp_plus,
            get_mbpp_plus_hash,
            load_solutions,
        )
        from evalplus.data.mbpp import mbpp_serialize_inputs
        from evalplus.eval import (
            PASS,
            compatible_eval_result,
            estimate_pass_at_k,
            untrusted_check,
        )
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import get_groundtruth, check_correctness


        assert samples is not None, "No samples provided"

        n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

        if os.path.isdir(samples):
            result_path = os.path.join(samples, "eval_results.json")
        else:
            assert samples.endswith(".jsonl")
            result_path = samples.replace(".jsonl", "_eval_results.json")

        if os.path.isfile(result_path) and not i_just_wanna_run:
            print(f"Load from previous results from {result_path}")
            with open(result_path, "r") as f:
                results = json.load(f)

            results = compatible_eval_result(results)
        else:
            if dataset == "humaneval":
                problems = get_human_eval_plus(
                    mini=mini, noextreme=noextreme, version=version
                )
                dataset_hash = get_human_eval_plus_hash(
                    mini=mini, noextreme=noextreme, version=version
                )
                expected_output = get_groundtruth(problems, dataset_hash, [])
            elif dataset == "mbpp":
                problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
                dataset_hash = get_mbpp_plus_hash(
                    mini=mini, noextreme=noextreme, version=version
                )
                expected_output = get_groundtruth(
                    problems,
                    dataset_hash,
                    MBPP_OUTPUT_NOT_NONE_TASKS,
                )

            results = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "hash": dataset_hash,
                "eval": {},
            }

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                completion_id = Counter()
                n_samples = 0
                eval_results = defaultdict(list)  # task_id ->
                remainings = set()

                print("Reading samples...")
                for sample in tqdm(load_solutions(samples)):
                    task_id = sample["task_id"]
                    if task_id not in problems:
                        warn(
                            f"Task {task_id} is found in the samples but not found in the dataset"
                        )
                        continue
                    solution = (
                        sample["solution"]
                        if "solution" in sample
                        else problems[task_id]["prompt"] + sample["completion"]
                    )
                    remainings.add(sample["_identifier"])
                    args = (
                        dataset,
                        completion_id[task_id],
                        problems[task_id],
                        solution,
                        expected_output[task_id],
                        base_only,
                        not test_details,  # fast_check
                        sample["_identifier"],
                        min_time_limit,
                        gt_time_limit_factor,
                    )
                    futures.append(executor.submit(check_correctness, *args))
                    completion_id[task_id] += 1
                    n_samples += 1

                assert n_samples == len(remainings), "Missing problems in unfinished"

                my_problems = dict()
                for task_id in completion_id.keys():
                    my_problems[task_id] = problems[task_id]
                problems = my_problems

                assert len(completion_id) == len(problems), "Missing problems in samples"

                def stucking_checker():
                    while remainings:
                        last_size = len(remainings)
                        time.sleep(20)
                        if last_size != len(remainings) or len(remainings) == 0:
                            continue
                        # Potential stucking
                        warn("No samples had finished testing in the last 20s")
                        warn(f"{len(remainings)} samples to be tested: {remainings}")

                threading.Thread(target=stucking_checker).start()

                for future in tqdm(as_completed(futures), total=n_samples):
                    result = future.result()
                    remainings.remove(result["_identifier"])
                    eval_results[result["task_id"]].append(result)

            # sort the results for each problem by completion_id
            for task_id, task_results in eval_results.items():
                task_results.sort(key=lambda x: x["completion_id"])
                results["eval"][task_id] = []
                for res in task_results:

                    def get_failed_tests(stat, details, inputs) -> List[Any]:
                        if stat == PASS or not details:
                            return []

                        if test_details:
                            return [
                                inputs[i] for i in range(len(details)) if not details[i]
                            ]

                        # else => simply return the only and the last fail test
                        return [inputs[len(details) - 1]]

                    base_stat, base_details = res["base"]
                    base_fail_tests = get_failed_tests(
                        base_stat, base_details, problems[task_id]["base_input"]
                    )

                    # initialize plus tests
                    plus_stat = None
                    plus_fail_tests = []

                    # with plus tests
                    if not base_only:
                        plus_stat, plus_details = res["plus"]
                        plus_fail_tests = get_failed_tests(
                            plus_stat, plus_details, problems[task_id]["plus_input"]
                        )

                    if dataset == "mbpp":
                        base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                        plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

                    results["eval"][task_id].append(
                        {
                            "task_id": task_id,
                            "solution": res["solution"],
                            "base_status": base_stat,
                            "plus_status": plus_stat,
                            "base_fail_tests": base_fail_tests,
                            "plus_fail_tests": plus_fail_tests,
                        }
                    )

        # Calculate pass@k.
        total = np.array([len(r) for r in results["eval"].values()])
        base_correct = []
        new_correct = []

        for res in results["eval"].values():
            bc = sum([r["base_status"] == PASS for r in res])
            base_correct.append(bc)
            if not base_only:
                new_correct.append(
                    sum(
                        [
                            res[i]["base_status"] == res[i]["plus_status"] == PASS
                            for i in range(len(res))
                        ]
                    )
                )
        base_correct = np.array(base_correct)

        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
            for k in [1, 10, 100]
            if total.min() >= k
        }
        cprint(f"{dataset} (base tests)", "red")
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "red")
        results["pass_at_k"] = {"base": pass_at_k}

        if new_correct:
            cprint(f"{dataset}+ (base + extra tests)", "green")
            pass_at_k = {
                f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
                for k in [1, 10, 100]
                if (total >= k).all()
            }
            for k, v in pass_at_k.items():
                cprint(f"{k}:\t{v:.3f}", "green")
            results["pass_at_k"]["plus"] = pass_at_k

        # save results
        if os.path.isfile(result_path) and i_just_wanna_run:
            decision = ""
            while decision.lower() not in ["y", "n"]:
                print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
                decision = input()

            if decision.lower() == "y":
                # mv the file to a backup
                new_path = result_path + ".bak"
                while os.path.isfile(new_path):
                    new_path += ".bak"
                os.rename(result_path, new_path)
                print(f"Backup {result_path} to {new_path}")

        if not os.path.isfile(result_path):
            with open(result_path, "w") as f:
                json.dump(results, f)
        return results