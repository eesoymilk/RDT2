#!/usr/bin/env python
# coding=utf-8
"""
UCVLA Stage 1 training: train user_bias + bias_proj while keeping the base RDT frozen.
Forked from rdt/train.py — key differences:
  - Wraps RDTRunner with UCVLARDTRunner after loading
  - freeze_base() locks all base weights
  - Optimizer only covers trainable_parameters() (user_bias + bias_proj)
  - Training loop passes user_id to compute_loss()
  - Checkpointing saves ucvla_weights.pt (trainable params only)
  - log_sample_res is skipped (does not support user_id yet)
"""
import logging
import math
import os
import socket
from pathlib import Path
from functools import partial

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DataLoaderConfiguration, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.normalizer import LinearNormalizer
from models.rdt_runner import RDTRunner
from models.ucvla.ucvla_runner import UCVLARDTRunner
from rdt.dataset import get_instructions_and_blended_train_dataset, get_val_dataset, collate_fn


if is_wandb_available():
    import wandb


def train(args, logger):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        padding_side="left",
        use_fast=True,
    )
    vision_language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        torch_dtype=weight_dtype,
        attn_implementation="flash_attention_2",
        device_map=accelerator.device,
    )
    vision_language_model.eval()

    vision_encoder = None

    if isinstance(config["model"]["selected_layers"], list):
        assert len(config["model"]["selected_layers"]) == config["model"]["rdt"]["depth"]
    elif not isinstance(config["model"]["selected_layers"], int):
        raise ValueError(f"selected_layers must be int or list, got {config['model']['selected_layers']}")

    # Load finetuned RDT checkpoint and wrap with UCVLA
    assert args.pretrained_model_name_or_path is not None, \
        "--pretrained_model_name_or_path is required for UCVLA Stage 1 (point to finetuned RDT checkpoint)"
    logger.info(f"Loading base RDT from {args.pretrained_model_name_or_path}")
    base_rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)

    ucvla_runner = UCVLARDTRunner(base_rdt, n_users=args.n_users, d_bias=args.d_bias)
    ucvla_runner.freeze_base()

    trainable = ucvla_runner.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"UCVLA trainable params: {n_trainable:,}  (user_bias + bias_proj, base frozen)")

    # Checkpointing: save only trainable params
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(ucvla_runner)
            torch.save(
                {
                    "user_bias": unwrapped.ucvla_model.user_bias.state_dict(),
                    "bias_proj": unwrapped.ucvla_model.bias_proj.state_dict(),
                    "n_users": args.n_users,
                    "d_bias": args.d_bias,
                },
                os.path.join(output_dir, "ucvla_weights.pt"),
            )
        # Prevent Accelerate's default saver from also saving the full model
        while len(weights) > 0:
            weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    assert args.webdataset_config is not None, "--webdataset_config is required"
    with open(args.webdataset_config, "r") as f:
        hostname = socket.gethostname()
        dataset_config_str = f.read().format(hostname=hostname)
        wds_config = yaml.safe_load(dataset_config_str)

    instructions, train_dataset = get_instructions_and_blended_train_dataset(wds_config)

    train_collate_fn = partial(
        collate_fn,
        processor=processor,
        instructions=instructions,
        image_corruption=args.image_aug,
        state_dim=config["common"]["state_dim"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=train_collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = get_val_dataset(wds_config["shards_dir"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        collate_fn=train_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    normalizer = LinearNormalizer.load(wds_config["kwargs"]["normalizer_path"])

    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    ucvla_runner, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        ucvla_runner, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if vision_language_model is not None:
        vision_language_model.to(accelerator.device, dtype=weight_dtype)

    if hasattr(train_dataset, "__len__") and len(train_dataset) > 0:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    else:
        num_update_steps_per_epoch = args.max_train_steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(os.getenv("WANDB_PROJECT", "rdt-2-ucvla"), config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running UCVLA Stage 1 Training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  n_users = {args.n_users}, d_bias = {args.d_bias}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            ckpt = torch.load(os.path.join(args.output_dir, path, "ucvla_weights.pt"), map_location="cpu")
            unwrapped = accelerator.unwrap_model(ucvla_runner)
            unwrapped.ucvla_model.user_bias.load_state_dict(ckpt["user_bias"])
            unwrapped.ucvla_model.bias_proj.load_state_dict(ckpt["bias_proj"])
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    def run_cross_user_eval(step):
        """Cross-user confusion matrix: correct bias should give lower action error."""
        ucvla_runner.eval()
        # errors[(uid_true, uid_pred)] = list of per-sample MSE values
        errors = {(ut, up): [] for ut in range(args.n_users) for up in range(args.n_users)}

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_actions = val_batch["actions"]
                val_nsamples = normalizer["action"].normalize(val_actions).to(
                    dtype=weight_dtype, device=accelerator.device
                )
                val_states = val_batch["states"].to(dtype=weight_dtype, device=accelerator.device)
                uid_true = val_batch["user_id"].to(accelerator.device)

                lang_attn_mask = val_batch["vision_language_model_inputs"]["attention_mask"].to(dtype=torch.bool)
                outputs = vision_language_model(
                    **val_batch["vision_language_model_inputs"],
                    use_cache=True,
                )
                selected_layers = config["model"]["selected_layers"]
                if isinstance(selected_layers, list):
                    vlang_kv_cache = [outputs.past_key_values[i] for i in selected_layers]
                else:
                    vlang_kv_cache = [outputs.past_key_values[selected_layers]]

                for uid_pred in range(args.n_users):
                    uid_tensor = torch.full_like(uid_true, uid_pred)
                    with torch.autocast("cuda", dtype=weight_dtype):
                        pred = accelerator.unwrap_model(ucvla_runner).predict_action(
                            user_id=uid_tensor,
                            state_tokens=val_states,
                            lang_kv_cache=vlang_kv_cache,
                            lang_attn_mask=lang_attn_mask,
                        )
                    # per-sample MSE
                    err = torch.nn.functional.mse_loss(pred, val_nsamples, reduction="none").mean(dim=[1, 2])
                    for i, ut in enumerate(uid_true.tolist()):
                        errors[(ut, uid_pred)].append(err[i].item())

        if accelerator.is_main_process:
            log_dict = {}
            header = "val cross-user confusion (rows=true, cols=pred bias):"
            rows = []
            for ut in range(args.n_users):
                row = []
                for up in range(args.n_users):
                    vals = errors[(ut, up)]
                    mean_err = sum(vals) / len(vals) if vals else float("nan")
                    row.append(f"{mean_err:.4f}")
                    log_dict[f"val/err_true{ut}_pred{up}"] = mean_err
                rows.append(f"  user_{ut}: [{', '.join(row)}]")
            logger.info(header + "\n" + "\n".join(rows))
            accelerator.log(log_dict, step=step)

        ucvla_runner.train()

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        ucvla_runner.train()

        for batch in train_dataloader:
            with accelerator.accumulate(ucvla_runner):
                actions = batch["actions"]
                nsamples = normalizer["action"].normalize(actions).to(
                    dtype=weight_dtype, device=accelerator.device
                )
                states = batch["states"].to(dtype=weight_dtype, device=accelerator.device)
                user_id = batch["user_id"].to(accelerator.device)

                with torch.no_grad():
                    lang_attn_mask = batch["vision_language_model_inputs"]["attention_mask"].to(dtype=torch.bool)
                    outputs = vision_language_model(
                        **batch["vision_language_model_inputs"],
                        use_cache=True,
                    )
                    selected_layers = config["model"]["selected_layers"]
                    if isinstance(selected_layers, list):
                        vlang_kv_cache = [outputs.past_key_values[i] for i in selected_layers]
                    else:
                        vlang_kv_cache = [outputs.past_key_values[selected_layers]]

                loss = ucvla_runner(
                    action_gt=nsamples,
                    user_id=user_id,
                    state_tokens=states,
                    lang_kv_cache=vlang_kv_cache,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=None,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                    run_cross_user_eval(global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if accelerator.is_main_process and global_step % 100 == 0:
                unwrapped = accelerator.unwrap_model(ucvla_runner)
                w = unwrapped.ucvla_model.user_bias.weight
                for i in range(args.n_users):
                    logs[f"bias_norm/user_{i}"] = w[i].norm().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save final weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(ucvla_runner)
        torch.save(
            {
                "user_bias": unwrapped.ucvla_model.user_bias.state_dict(),
                "bias_proj": unwrapped.ucvla_model.bias_proj.state_dict(),
                "n_users": args.n_users,
                "d_bias": args.d_bias,
            },
            os.path.join(args.output_dir, "ucvla_weights.pt"),
        )
        logger.info(f"Saved final UCVLA weights to {args.output_dir}/ucvla_weights.pt")

    accelerator.end_training()
