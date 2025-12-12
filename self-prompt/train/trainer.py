# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import os
from typing import Optional
from transformers import Trainer

import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

import importlib.metadata

def get_transformers_version():
    return importlib.metadata.version("transformers")

version = get_transformers_version()
print(version)
if not version == "4.30.2":
    from transformers.trainer import (
        is_torch_xla_available
    )
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
else:
    from transformers.trainer import (
        is_torch_tpu_available
    )
    if is_torch_tpu_available(check_device=False):
        import torch_xla.core.xla_model as xm

from typing import Dict, Optional

import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PrefixTrainer(Trainer):
    def __init__(self, *args, save_changed=False, **kwargs):
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)
        self.my_train_loss = []
        self.my_train_loss_step = []
        self.epoch = 0
        if version == "4.30.2":
            Trainer._maybe_log_save_evaluate = self._maybe_log_save_evaluate_old
        else:
            Trainer._maybe_log_save_evaluate = self._maybe_log_save_evaluate_new

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.save_changed:
                print("Saving PrefixEncoder")
                state_dict = self.model.state_dict()
                filtered_state_dict = {}
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]
                self.model.save_pretrained(output_dir, state_dict=filtered_state_dict)
            else:
                print("Saving the whole model")
                self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def show_loss(self, loss_list, x_name='step'):
        y_train_loss = loss_list  # loss值，即y轴
        x_train_loss = range(1, len(y_train_loss)+1)  # loss的数量，即x轴

        plt.figure()

        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel(x_name)  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")

        plt.legend()
        plt.title('Loss curve')
        if not os.path.exists('./pic'):
            os.mkdir('./pic')
        plt.savefig('./pic/loss.png')
        plt.close()

    def _maybe_log_save_evaluate_old(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            raw_model = unwrap_model(model)
            dataset_name = getattr(raw_model.config, 'dataset_name', 'default')
            model_name = getattr(raw_model.config, 'model_name_for_pt', 'default')
            
            pt_dir = os.path.join('./pt_file', f"{model_name}_{dataset_name}")
            os.makedirs(pt_dir, exist_ok=True)

            torch.save(logs["loss"], os.path.join(pt_dir, 'epoch_loss.pt'))
            torch.save(epoch, os.path.join(pt_dir, 'epoch.pt'))

            self.my_train_loss_step.append(logs["loss"])
            if self.epoch != epoch:
                self.my_train_loss.append(sum(self.my_train_loss_step) / len(self.my_train_loss_step))
                self.show_loss(self.my_train_loss, 'epoch')
                self.my_train_loss_step = []
                self.epoch = epoch

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _maybe_log_save_evaluate_new(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval=None,*args,**kwargs):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            raw_model = unwrap_model(model)
            dataset_name = getattr(raw_model.config, 'dataset_name', 'default')
            model_name = getattr(raw_model.config, 'model_name_for_pt', 'default')
            
            pt_dir = os.path.join('./pt_file', f"{model_name}_{dataset_name}")
            os.makedirs(pt_dir, exist_ok=True)

            torch.save(logs["loss"], os.path.join(pt_dir, 'epoch_loss.pt'))
            torch.save(epoch, os.path.join(pt_dir, 'epoch.pt'))

            self.my_train_loss_step.append(logs["loss"])
            if self.epoch != epoch:
                self.my_train_loss.append(sum(self.my_train_loss_step) / len(self.my_train_loss_step))
                self.show_loss(self.my_train_loss, 'epoch')
                self.my_train_loss_step = []
                self.epoch = epoch

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
