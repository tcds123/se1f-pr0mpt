import os.path

from transformers import AutoTokenizer
from transformers.utils import logging
from .modeling_phi3_small import Phi3SmallModel, Phi3SmallForCausalLM, min_value_of_dtype

from torch.nn import CrossEntropyLoss
import torch
import random
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from my_model.my_embedding import Embedding

logger = logging.get_logger(__name__)

random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = AutoTokenizer.from_pretrained('/data/team/zongwx1/llm_models/Phi-3-small-8k-instruct', trust_remote_code=True)


class MyPhi3Model(Phi3SmallModel):
    def __init__(self, config):
        super().__init__(config)

        print("************** My PhiModel")
        self.is_train_mode = False
        if hasattr(config, 'step'):
            self.is_train_mode = True
            self.embed_tokens = Embedding(config, self.embed_tokens, TOKENIZER)
            for name, param in self.named_parameters():
                if 'word_embeddings' in name:
                    param.requires_grad = True
                    continue
                param.requires_grad_(False)

            for name, param in self.named_parameters():
                if param.requires_grad is True:
                    print(name)


class MyPhi3ForCausalLM(Phi3SmallForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MyPhi3Model(config)

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.mup_width_multiplier:
            logits = logits / self.mup_width_multiplier
        logits = logits.masked_fill(self.dummy_tokens_mask, min_value_of_dtype(logits.dtype))

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
        if not os.path.exists('./pt_file'):
            os.mkdir('./pt_file')
        torch.save(value, './pt_file/' + name + '.pt')
