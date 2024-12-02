import os.path

from transformers import AutoTokenizer

from .modeling_chatglm import (
    ChatGLMModel,
    ChatGLMForConditionalGeneration,
)

from torch import nn
from torch.nn import CrossEntropyLoss
import torch
import random
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
import copy
from my_embedding import Embedding as MyEmbedding

logger = logging.get_logger(__name__)

random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = AutoTokenizer.from_pretrained('/data/team/zongwx1/llm_models/chatglm3-6b', trust_remote_code=True)
MODEL_NAME = 'chatglm3-6b'

"""
from torch.nn.utils import skip_init
def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

class Embedding(torch.nn.Module):

    def __init__(self, config, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection
        self.is_train_sys_prompt_mode = False

        self.padded_vocab_size = config.padded_vocab_size
        self.torch_dtype = config.torch_dtype
        self.device = device
        self.tokenizer = TOKENIZER
        self.model_name = MODEL_NAME

        if hasattr(config, 'step'):
            self.is_train_sys_prompt_mode = True
            self.device = 'cuda'
            self.w_emb = None
            self.sys_prompt_len = config.sys_prompt_len
            self.sys_prompt_is_list = config.sys_prompt_is_list
            if hasattr(config, 'extra_len'):
                self.extra_len = config.extra_len
            else:
                self.extra_len = 0
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
            self.record_epoch = [1, 2, 3, 5, 10, 15, 20]
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


    def forward(self, input_ids):
        if self.w_emb is None:
            self.w_emb = nn.Embedding(
                self.padded_vocab_size,
                self.hidden_size,
                dtype=self.torch_dtype,
                device=self.device,
                _weight=copy.deepcopy(self.word_embeddings.weight)
            )
        if self.extra_len > 0 and self.extra_ids is None:
            self.extra_ids = self.initialize_extra_ids(input_ids, self.extra_len).to(self.device)

        embeddings = self.my_train_sys_emb(input_ids)

        return embeddings

    def my_train_sys_emb(self, input_ids):
        w_embeddings = self.w_emb

        input_embedding = w_embeddings(input_ids[:, self.skip_num_front + self.skip_num_back:])
        sys_tensor = input_ids[:, :self.sp_token_num[0] + self.sys_prompt_len]

        if self.extra_len > 0:
            sys_tensor = torch.cat((sys_tensor, self.extra_ids, input_ids[:, self.skip_num_front:self.skip_num_front+self.skip_num_back]), dim=1)
        else:
            sys_tensor = torch.cat((sys_tensor, input_ids[:, self.skip_num_front:self.skip_num_front + self.skip_num_back]),dim=1)
        sys_emb = self.word_embeddings(sys_tensor)

        if self.skip_count < 1:
            label = self.get_shift_labels()
            self.update_new_label_set(label)
        self.skip_count -= 1
        if self.ignore_step <= 0:
            self.get_change_most_input_ids(w_embeddings.weight.detach(), self.word_embeddings.weight.detach())
        self.ignore_step -= 1

        if self.skip <= 1:
            self.print_input_ids(sys_emb, w_emb=w_embeddings)
            self.skip = self.base_skip + 1
        self.skip -= 1

        embeddings = torch.cat((sys_emb, input_embedding), dim=1)

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
        print("torch.topk(ori_emb_table[0], 10): ", torch.topk(ori_emb_table[:3], 10))
        print("torch.topk(cur_emb_table[0], 10): ", torch.topk(cur_emb_table[:3], 10))
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
                            dynamic_expand=False):
        epoch = self.get_epoch()
        if dynamic_expand:
            topk = epoch
        diff_emb_table = cur_emb_table - ori_emb_table
        sum_diff_emb_table = torch.sum(abs(diff_emb_table), dim=1).tolist()
        print(sorted(sum_diff_emb_table, reverse=True)[:10])
        new_sum_diff_emb_dict = {}
        if not self.use_all:
            for ids in input_ids:
                if ids in self.weight_dict:
                    new_sum_diff_emb_dict[ids] = self.weight_dict[ids] * sum_diff_emb_table[ids]
                else:
                    new_sum_diff_emb_dict[ids] = sum_diff_emb_table[ids]
        else:
            for ids in list(self.sample_ids[0]):
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
        for id in should_careful_structure:
            print(self.tokenizer.decode([id]), end=', ')
        if epoch >= self.record_epoch[0]:
            if not os.path.exists('./txt/' + self.model_name):
                os.mkdir('./txt/' + self.model_name)
            if not os.path.exists('./txt/' + self.model_name + '/' + self.dataset_name):
                os.mkdir('./txt/' + self.model_name + '/' + self.dataset_name)
            with open('./txt/' + self.model_name + '/' + self.dataset_name + '/' + str(self.record_epoch[0])+'.txt', 'w', encoding='utf-8') as f:
                for id in should_careful_structure:
                    print(self.tokenizer.decode([id]), end=', ', file=f)
            self.record_epoch = self.record_epoch[1:]
            f.close()

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

    def initialize_extra_ids(self, input_ids, extra_len: int):
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
"""


class MyChatGLMModel(ChatGLMModel):
    def __init__(self, config, device=None, empty_init=True):
        super().__init__(config, device, empty_init)

        print("************** My GlmModel")
        self.is_train_mode = False
        if hasattr(config, 'step'):
            self.is_train_mode = True
            self.embedding = MyEmbedding(config, self.embedding.word_embeddings, TOKENIZER, MODEL_NAME)
            for name, param in self.named_parameters():
                if 'my_word_embeddings' in name:
                    param.requires_grad = True
                    continue
                param.requires_grad_(False)

            for name, param in self.named_parameters():
                if param.requires_grad is True:
                    print(name)

        # self.my_init(config)

    def my_init(self, config):
        if not os.path.exists('./pt_file/glm3_embed_tokens.pt'):
            self.init_count = 1
            self.is_train_mode = False
            if hasattr(config, 'step'):
                self.is_train_mode = True
                print(torch.topk(self.embedding.my_word_embeddings.weight[:3], 10))
                self.my_embedding = MyEmbedding(config, self.embedding.my_word_embeddings, TOKENIZER, MODEL_NAME)
                for name, param in self.named_parameters():
                    if 'word_embeddings' in name:
                        param.requires_grad = True
                        continue
                    param.requires_grad_(False)

                for name, param in self.named_parameters():
                    if param.requires_grad is True:
                        print(name)
        else:
            self.init_count = 0
            self.is_train_mode = False
            if hasattr(config, 'step'):
                self.is_train_mode = True
                self.w_embed_tokens = torch.load('./pt_file/glm3_embed_tokens.pt')
                self.padding_idx = config.pad_token_id
                self.vocab_size = config.vocab_size
                print(torch.topk(self.w_embed_tokens[:3], 10))
                self.embedding.my_word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx,
                                                                 _weight=self.w_embed_tokens)
                self.embedding = MyEmbedding(config, self.embedding.my_word_embeddings, TOKENIZER, MODEL_NAME)
                for name, param in self.named_parameters():
                    if 'word_embeddings' in name:
                        param.requires_grad = True
                        continue
                    param.requires_grad_(False)

                for name, param in self.named_parameters():
                    if param.requires_grad is True:
                        print(name)

    # def forward(
    #         self,
    #         input_ids,
    #         position_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.BoolTensor] = None,
    #         full_attention_mask: Optional[torch.BoolTensor] = None,
    #         past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         use_cache: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ):
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     batch_size, seq_length = input_ids.shape
    #
    #     if inputs_embeds is None:
    #         # if self.init_count > 0:
    #         #     print(torch.topk(self.embed_tokens.weight[:3], 10))
    #         #     torch.save(self.embedding.word_embeddings.weight, './pt_file/glm3_embed_tokens.pt')
    #         #     assert False
    #         inputs_embeds = self.embedding(input_ids)
    #
    #     # if self.pre_seq_len is not None:
    #     #     if past_key_values is None:
    #     #         past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
    #     #                                           dtype=inputs_embeds.dtype)
    #     #     if attention_mask is not None:
    #     #         attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
    #     #                                     attention_mask], dim=-1)
    #
    #     if full_attention_mask is None:
    #         if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
    #             full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
    #
    #     # Rotary positional embeddings
    #     rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    #     if position_ids is not None:
    #         rotary_pos_emb = rotary_pos_emb[position_ids]
    #     else:
    #         rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    #     rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
    #
    #     # Run encoder.
    #     hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
    #         inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
    #         kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    #     )
    #
    #     if not return_dict:
    #         return tuple(
    #             v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
    #
    #     return BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=presents,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attentions,
    #     )


class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self, config, empty_init=True, device=None):
        super().__init__(config)

        self.transformer = MyChatGLMModel(config, empty_init=empty_init, device=device)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            self.save_pt(shift_labels, "shift_labels")

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # print(shift_logits.view(-1, shift_logits.size(-1)).size())
            # print(shift_labels.view(-1).size())
            # assert False
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

            self.save_pt(loss, 'loss')

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def save_pt(self, value, name):
        if not os.path.exists('./pt_file'):
            os.mkdir('./pt_file')
        torch.save(value, './pt_file/' + name + '.pt')
