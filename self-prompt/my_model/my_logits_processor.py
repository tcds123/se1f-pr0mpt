
import torch
from transformers.generation.logits_process import LogitsProcessor


class SensitiveTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, weight_dict: dict, max_weight: float = 1.5):
        super().__init__()
        self.weight_dict = weight_dict
        self.max_weight = max_weight
        max_epoch = max(self.weight_dict.keys())
        self.epoch_per_weight = (max_weight - 1) / max_epoch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for epoch, ids_list in self.weight_dict.items():
            for ids in ids_list:
                scores[:, ids] *= (1 + epoch * self.epoch_per_weight)

        return scores
