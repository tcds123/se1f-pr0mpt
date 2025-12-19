import os.path
import sys
sys.path.append(os.path.dirname(__file__))

from transformers.models.qwen3.modeling_qwen3 import (
    QWEN3_START_DOCSTRING, # 注意变量名通常也会变成 QWEN3
    QWEN3_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,       # 这个变量名可能没变，或者需要检查
    Qwen3Config,
    Qwen3Model,
    Qwen3ForCausalLM
)
from transformers import AutoTokenizer,Qwen3Config
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    logging
)
from torch.nn import CrossEntropyLoss
import torch
import random
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from .my_embedding import Embedding

logger = logging.get_logger(__name__)

random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: 请确认此处路径是否为你的 Qwen3-8B 模型实际路径
TOKENIZER = AutoTokenizer.from_pretrained('/data/private/self-prompt/models/Qwen3-8B')
MODEL_NAME = 'qwen3_8b'


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class MyQwen3Model(Qwen3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`QWEN3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

        print("************** My Qwen3Model")
        self.is_train_mode = False
        if hasattr(config, 'step'):
            self.is_train_mode = True
            # 使用自定义 Embedding
            self.embed_tokens = Embedding(config, self.embed_tokens, TOKENIZER, MODEL_NAME)
            
            # 冻结除 word_embeddings 以外的参数
            for name, param in self.named_parameters():
                if 'word_embeddings' in name:
                    param.requires_grad = True
                    continue
                param.requires_grad_(False)

            # 打印需要梯度的参数名称进行确认
            for name, param in self.named_parameters():
                if param.requires_grad is True:
                    print(name)


class MyQwen3ForCausalLM(Qwen3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MyQwen3Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, QWEN3ForCausalLM

        >>> model = QWEN3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            self.save_pt(shift_labels, "shift_labels")

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            self.save_pt(loss, 'loss')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pt(self, value, name):
        # 获取唯一目录名，默认为 'default' 以防万一
        dataset_name = getattr(self.config, 'dataset_name', 'default')
        model_name = getattr(self.config, 'model_name_for_pt', 'default')
        
        # 构造专属目录： ./pt_file/qwen2_humaneval/
        dir_path = f'./pt_file/{model_name}_{dataset_name}'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        # 保存文件
        torch.save(value, os.path.join(dir_path, name + '.pt'))