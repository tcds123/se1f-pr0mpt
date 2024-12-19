# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device


init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import deepspeed_utils, sdxl_model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel

from sdxl_infer import image_generation
import shutil
from clip_score import calc_clip_score
import json
from PIL import Image


UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    assert (
        not args.weighted_captions
    ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = False
    train_text_encoder1 = True
    train_text_encoder2 = True

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 > 0
        train_text_encoder2 = lr_te2 > 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # TextEncoderの出力をキャッシュする
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            with torch.no_grad(), accelerator.autocast():
                train_dataset_group.cache_text_encoder_outputs(
                    (tokenizer1, tokenizer2),
                    (text_encoder1, text_encoder2),
                    accelerator.device,
                    None,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )
            accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        for name, parameters in text_encoder1.named_parameters():
            if 'token_embedding' in name:
                print(f'text_encoder_1 optim_param: {name}')
                params_to_optimize.append(
                    {"params": parameters, "lr": args.learning_rate_te1 or args.learning_rate})

        # params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})

    if train_text_encoder2:
        training_models.append(text_encoder2)
        for name, parameters in text_encoder2.named_parameters():
            if 'token_embedding' in name:
                print(f'text_encoder_2 optim_param: {name}')
                params_to_optimize.append(
                    {"params": parameters, "lr": args.learning_rate_te2 or args.learning_rate})

        # params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, [tokenizer1, tokenizer2], [text_encoder1, text_encoder2], unet
    )

    loss_recorder = train_util.LossRecorder()

    self_prompt1 = SelfPrompt(training_models[0], tokenizer1)
    self_prompt2 = SelfPrompt(training_models[1], tokenizer2)

    if args.eval:
        eval_func = Eval(args.dataset_name)

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        # if args.weighted_captions:
                        #     encoder_hidden_states = get_weighted_text_embeddings(
                        #         tokenizer,
                        #         text_encoder,
                        #         batch["captions"],
                        #         accelerator.device,
                        #         args.max_token_length // 75 if args.max_token_length else 1,
                        #         clip_skip=args.clip_skip,
                        #     )
                        # else:
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        # unwrap_model is fine for models not wrapped by accelerator
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                            args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                            accelerator=accelerator,
                        )
                else:
                    encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                    # # verify that the text encoder outputs are correct
                    # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
                    #     args.max_token_length,
                    #     batch["input_ids"].to(text_encoder1.device),
                    #     batch["input_ids2"].to(text_encoder1.device),
                    #     tokenizer1,
                    #     tokenizer2,
                    #     text_encoder1,
                    #     text_encoder2,
                    #     None if not args.full_fp16 else weight_dtype,
                    # )
                    # b_size = encoder_hidden_states1.shape[0]
                    # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # logger.info("text encoder outputs verified")

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                target = noise

                if (
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c)
                    if args.masked_loss:
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c)

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                return_neg = True
                emb_direct = False
                if emb_direct:
                    if return_neg:
                        prompt1, neg_prompt1 = self_prompt1.get_prompt_from_emb(input_ids1, loss, epoch, return_neg=True)
                        prompt2, neg_prompt2 = self_prompt2.get_prompt_from_emb(input_ids2, loss, epoch, return_neg=True)
                    else:
                        prompt1 = self_prompt1.get_prompt_from_emb(input_ids1, loss, epoch)
                        prompt2 = self_prompt2.get_prompt_from_emb(input_ids2, loss, epoch)
                else:
                    if return_neg:
                        prompt1, neg_prompt1 = self_prompt1.get_prompt(input_ids1, loss, epoch, return_neg=True)
                        prompt2, neg_prompt2 = self_prompt2.get_prompt(input_ids2, loss, epoch, return_neg=True)
                    else:
                        prompt1 = self_prompt1.get_prompt(input_ids1, loss, epoch)
                        prompt2 = self_prompt2.get_prompt(input_ids2, loss, epoch)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if args.eval and prompt1 is not None and prompt2 is not None:
                    if epoch and step % 10 == 0:
                        print("*" * 50, "Eval begin")
                        prompt1_str = decode_ids(prompt1, tokenizer1)
                        prompt2_str = decode_ids(prompt2, tokenizer2)
                        if return_neg:
                            neg_prompt1_str = decode_ids(neg_prompt1, tokenizer1)
                            neg_prompt2_str = decode_ids(neg_prompt2, tokenizer2)
                            eval_func.eval(prompt1_str, prompt2_str, step * (epoch + 1), args.seed, neg_prompt1_str, neg_prompt2_str)
                        else:
                            eval_func.eval(prompt1_str, prompt2_str, step * (epoch + 1), args.seed)
                        print("*" * 50, "Eval end")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2],
                    unet,
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.eval:
        best_prompt1 = eval_func.best_prompt1
        best_prompt2 = eval_func.best_prompt2
        print(f"best prompt1: {best_prompt1}")
        print(f"best prompt2: {best_prompt2}")
        best_neg_prompt1 = eval_func.best_neg_prompt1
        best_neg_prompt2 = eval_func.best_neg_prompt2
        print(f"best negative prompt1: {best_neg_prompt1}")
        print(f"best negative prompt2: {best_neg_prompt2}")
    else:
        decoded_prompt1 = decode_ids(prompt1, tokenizer1)
        decoded_prompt2 = decode_ids(prompt2, tokenizer2)
        best_prompt1 = decoded_prompt1
        best_prompt2 = decoded_prompt2
        print(f"best prompt1: {best_prompt1}")
        print(f"best prompt2: {best_prompt2}")

    if args.save_state or args.save_state_on_train_end:        
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    logger.info("Training finished.")
    return best_prompt1, best_prompt2, best_neg_prompt1, best_neg_prompt2


def decode_ids(ids, tokenizer):
    ids_str = ""
    for id in ids:
        ids_str += tokenizer.decode([id]) + ", "
    return ids_str


class Eval:
    def __init__(self, dataset_name):
        self.best_prompt1 = ""
        self.best_prompt2 = ""
        self.best_neg_prompt1 = ""
        self.best_neg_prompt2 = ""
        self.best_clip_score = float('-inf')
        self.dataset_path = './dataset/' + dataset_name + '/'
        val_folder = self.dataset_path + "val/"
        self.ori_folder = val_folder + "ori/"
        self.renew_dir(self.ori_folder)
        self.opt_folder = val_folder + "opt/"
        self.train_folder = self.dataset_path + "10_train/"
        self.train_data = os.listdir(self.train_folder)
        params = self.dataset_path + 'params.json'
        for data in self.train_data:
            if data.endswith('.txt'):
                with open(self.train_folder + data, 'r') as f:
                    self.base_prompt = f.readline()
                    self.index = data.split('.')[0]
            if data.endswith('.jpg'):
                with open(self.train_folder + data, 'rb') as f:
                    img = Image.open(f)
                    img.save(self.ori_folder + data)

        with open(params, 'r') as f:
            param = json.load(f)
            self.height = param[self.index]['height'] - param[self.index]['height'] % 8
            self.width = param[self.index]['width'] - param[self.index]['width'] % 8
        clip_model_path = "./clip-ViT-B-32/ViT-B-32.pt"
        self.clip_score_func = calc_clip_score.ClipScore(clip_model_path, 'cuda')

    def eval(self, prompt1, prompt2, step, seed, neg_prompt1=None, neg_prompt2=None):
        extra_prompt = prompt1 + prompt2
        if neg_prompt1 and neg_prompt2:
            negative_prompt = neg_prompt1 + neg_prompt2
            clip_score = self.calc_score(extra_prompt, seed, negative_prompt=negative_prompt)
            if clip_score > self.best_clip_score:
                self.best_prompt1 = prompt1
                self.best_prompt2 = prompt2
                self.best_neg_prompt1 = neg_prompt1
                self.best_neg_prompt2 = neg_prompt2
                self.best_clip_score = clip_score
            print(
                f"step: {step}, best_clip_score: {self.best_clip_score}, best extra prompt: {self.best_prompt1 + self.best_prompt2}, best negative prompt: {self.best_neg_prompt1 + self.best_neg_prompt2}")
        else:
            clip_score = self.calc_score(extra_prompt, seed)

            if clip_score > self.best_clip_score:
                self.best_prompt1 = prompt1
                self.best_prompt2 = prompt2
                self.best_clip_score = clip_score
            print(f"step: {step}, best_clip_score: {self.best_clip_score}, best extra prompt: {self.best_prompt1 + self.best_prompt2}")

    def calc_score(self, extra_prompt, seed, negative_prompt=None):
        self.renew_dir(self.opt_folder)

        prompt = self.base_prompt + extra_prompt
        # prompt = extra_prompt
        image_generation(prompt,
                         save_path=self.opt_folder + self.index + '.jpg',
                         height=self.height,
                         width=self.width,
                         seed=seed,
                         negative_prompt=negative_prompt,)
        real_path = self.ori_folder
        fake_path = self.opt_folder
        clip_score = self.clip_score_func.call_main(real_path, fake_path)
        return clip_score

    def renew_dir(self, filepath):
        '''
        如果文件夹不存在就创建，如果文件存在就清空！
        '''
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:
            shutil.rmtree(filepath)
            os.mkdir(filepath)


class SelfPrompt:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.grad = None
        self.pre_loss = None
        self.weight_dict = {}
        self.new_sum_diff_emb_dict = {}
        self.should_careful_list = []
        self.should_careful_list_neg = []
        self.sp_token = [267, 49406, 49407]  # [, , <|startoftext|>, <|endoftext|>]

    def update_grad(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if 'token_embedding' in name:
                    grad = param.grad.clone().detach()
                    if self.grad is None:
                        self.grad = grad
                    else:
                        self.grad += grad
                    print('token_embedding grad has been updated.')

    def update_weight_table(self, input_ids, cur_loss):
        sum_diff_emb_table = torch.sum(abs(self.grad), dim=1).tolist()
        print(sorted(sum_diff_emb_table, reverse=True)[:10])
        sample_list = [i for i in range(len(sum_diff_emb_table))]
        print(f"sample_list len = {len(sample_list)}")

        weight = 1 - ((cur_loss - self.pre_loss) / max(cur_loss, self.pre_loss)) # 分母加个abs

        # for ids in sample_list:
        for ids in input_ids[0][0]:
            # print("========", ids)
            ids = ids.item()
            # print("**********", ids)
            if ids in self.sp_token:
                continue
            if ids in self.weight_dict:
                self.weight_dict[ids] *= weight  # 更新权重
                self.new_sum_diff_emb_dict[ids] += sum_diff_emb_table[ids] * self.weight_dict[ids]  # 更新梯度
            else:
                self.weight_dict[ids] = weight
                self.new_sum_diff_emb_dict[ids] = sum_diff_emb_table[ids]
        # print(self.weight_dict)
        print("weight: ", round(weight, 4))

    def sample_topk_ids(self, sorted_dict, topk, reverse=False):
        target_list = []
        if reverse:
            sorted_dict.reverse()
        for tup in sorted_dict:
            if len(target_list) < topk and tup[0] not in self.sp_token:
                target_list.append(tup)

        return target_list

    def new_update_weight_table(self, input_ids, cur_loss):
        sum_diff_emb_table = torch.sum(abs(self.grad), dim=1).tolist()
        print(sorted(sum_diff_emb_table, reverse=True)[:10])
        sample_list = [i for i in range(len(sum_diff_emb_table))]
        print(f"sample_list len = {len(sample_list)}")

        weight = 1 - ((cur_loss - self.pre_loss) / max(cur_loss, self.pre_loss))

        for ids in sample_list:
        # for ids in input_ids[0][0]:
            # print("========", ids)
            # ids = ids.item()
            # print("**********", ids)
            if ids in self.sp_token:
                continue
            if ids in self.weight_dict:
                self.weight_dict[ids] *= weight  # 更新权重
                self.new_sum_diff_emb_dict[ids] = sum_diff_emb_table[ids]  # 更新梯度
            else:
                self.weight_dict[ids] = weight
                self.new_sum_diff_emb_dict[ids] = sum_diff_emb_table[ids]
        # print(self.weight_dict)
        print("weight: ", round(weight, 4))

    def get_prompt_from_emb(self, input_ids, loss, epoch, topk=16, return_neg=False):
        loss = loss.clone().detach().item()
        if self.grad is None:
            self.update_grad()
            self.pre_loss = loss
            return None, None
        else:
            self.update_grad()
            if loss > self.pre_loss:
                print("***************** 检测到loss上升 *****************")

            self.new_update_weight_table(input_ids, loss)
            self.pre_loss = loss
        # if epoch < 1:
        #     return None, None
        # else:
        #     topk = min(epoch, topk)

        sorted_dict = sorted(self.new_sum_diff_emb_dict.items(), key=lambda d: d[1], reverse=True)
        # print(self.new_sum_diff_emb_dict.items())
        # print(self.weight_dict.items())
        # sorted_dict = sorted(self.weight_dict.items(), key=lambda d: d[1], reverse=True)
        max_values = self.sample_topk_ids(sorted_dict, topk)
        max_indices = []
        for dict in max_values:
            max_indices.append(dict[0])

        for max_index in max_indices:
            print([max_index], end=" | ")
        print("")
        for max_index in max_indices:
            print([self.tokenizer.decode([max_index])], end=" | ")
        print("")

        self.should_careful_list = max_indices

        print(self.should_careful_list)
        print("*********** self.should_careful_list: ", end='')
        for id in self.should_careful_list:
            print(self.tokenizer.decode([id]), end=', ')

        print("\n")
        if return_neg:
            min_values = self.sample_topk_ids(sorted_dict, topk, reverse=True)
            min_indices = []
            for dict in min_values:
                min_indices.append(dict[0])

            for min_index in min_indices:
                if min_index not in self.should_careful_list_neg and min_index not in self.sp_token:
                    self.should_careful_list_neg.append(min_index)

            neg_str = decode_ids(self.should_careful_list_neg, self.tokenizer)
            print(f"*********** self.should_careful_list(neg): {neg_str}")
            return self.should_careful_list, self.should_careful_list_neg

        return self.should_careful_list

    def get_prompt(self, input_ids, loss, epoch, topk=5, return_neg=False):
        # input_ids = self.clean_input_ids(input_ids)
        # print(f"input_ids: {input_ids}")
        loss = loss.clone().detach().item()
        if self.grad is None:
            self.update_grad()
            self.pre_loss = loss
            return None, None
        else:
            self.update_grad()
            if loss > self.pre_loss:
                print("***************** 检测到loss上升 *****************")

            self.update_weight_table(input_ids, loss)
            self.pre_loss = loss
        if epoch < 1:
            return None, None
        else:
            topk = min(epoch, topk)

        sorted_dict = sorted(self.new_sum_diff_emb_dict.items(), key=lambda d: d[1], reverse=True)
        print(self.new_sum_diff_emb_dict.items())
        # print(self.weight_dict.items())
        # sorted_dict = sorted(self.weight_dict.items(), key=lambda d: d[1], reverse=True)
        max_values = self.sample_topk_ids(sorted_dict, topk)
        max_indices = []
        for dict in max_values:
            max_indices.append(dict[0])

        for max_index in max_indices:
            print([max_index])
            print([self.tokenizer.decode([max_index])])
            print("----------")

            if max_index not in self.should_careful_list and max_index not in self.sp_token:
                self.should_careful_list.append(max_index)

        print(self.should_careful_list)
        print("*********** self.should_careful_list: ", end='')
        for id in self.should_careful_list:
            print(self.tokenizer.decode([id]), end=', ')

        print("\n")
        if return_neg:
            min_values = self.sample_topk_ids(sorted_dict, topk, reverse=True)
            min_indices = []
            for dict in min_values:
                min_indices.append(dict[0])

            for min_index in min_indices:
                if min_index not in self.should_careful_list_neg and min_index not in self.sp_token:
                    self.should_careful_list_neg.append(min_index)

            neg_str = decode_ids(self.should_careful_list_neg, self.tokenizer)
            print(f"*********** self.should_careful_list(neg): {neg_str}")
            return self.should_careful_list, self.should_careful_list_neg

        return self.should_careful_list


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--original_config",
        type=str,
        default=None,
        help="config of model.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=True,
        help=f"use eval or not, default is False",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="laion",
        help=f"name of dataset",
    )
    return parser


def do_single_train(parser):
    args = init_args(parser)

    train_data_dir = args.train_data_dir
    train_path = train_data_dir + '/10_train'
    # image_path = train_data_dir + '/image'
    image_path = train_data_dir + '/val_image_100'
    # text_path = train_data_dir + '/text'
    text_path = train_data_dir + '/val_text_100'
    image_list = os.listdir(image_path)
    for image in image_list:
        try:
            args = init_args(parser)

            index = image.split('.')[0]
            img_path = image_path + '/' + image
            txt_path = text_path + '/' + index + '.txt'
            # 清空train_path
            if os.path.exists(train_path):
                shutil.rmtree(train_path)
            os.makedirs(train_path, exist_ok=True)
            with open(txt_path, 'r') as f:
                text = f.read()
            with open(img_path, 'rb') as f:
                image_data = f.read()
            with open(train_path + '/' + index + '.txt', 'w') as f:
                f.write(text)
            with open(train_path + '/' + image, 'wb') as f:
                f.write(image_data)
            prompt1, prompt2, neg_prompt1, neg_prompt2 = train(args)
            if prompt1 and prompt2:
                prompt_path = train_data_dir + '/extra_prompt'
                prompt1_path = prompt_path + '/' + index + '_1.txt'
                prompt2_path = prompt_path + '/' + index + '_2.txt'
                with open(prompt1_path, 'w') as f:
                    f.write(prompt1)
                with open(prompt2_path, 'w') as f:
                    f.write(prompt2)
            if neg_prompt1 and neg_prompt2:
                prompt_path = train_data_dir + '/extra_prompt'
                neg_prompt1_path = prompt_path + '/neg_' + index + '_1.txt'
                neg_prompt2_path = prompt_path + '/neg_' + index + '_2.txt'
                with open(neg_prompt1_path, 'w') as f:
                    f.write(neg_prompt1)
                with open(neg_prompt2_path, 'w') as f:
                    f.write(neg_prompt2)
        except Exception as e:
            print(f'{image} train failed caused by {e}')
            continue
    print(f'all {len(image_list)} images train finished!')


def init_args(parser):
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    return args


if __name__ == "__main__":
    parser = setup_parser()

    do_single_train(parser)

    # args = parser.parse_args()
    # train_util.verify_command_line_training_args(args)
    # args = train_util.read_config_from_file(args, parser)

    # train(args)

