import argparse
import os
from rdt.train_ucvla_stage1 import train

from accelerate.logging import get_logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="UCVLA Stage 1: train per-user bias on frozen RDT backbone.")
    parser.add_argument("--config_path", type=str, default="configs/base.yaml")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--pretrained_vision_language_model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the finetuned RDT checkpoint directory (output of finetune_rdt.sh).",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        required=True,
        help="Number of distinct users (size of the user_bias embedding table).",
    )
    parser.add_argument(
        "--d_bias",
        type=int,
        default=64,
        help="Per-user bias dimension. Default: 64.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/ucvla/stage1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--num_sample_batches", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_period", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hdf5_dir", type=str, default=None)
    parser.add_argument("--webdataset_config", type=str, default=None)
    parser.add_argument("--image_aug", action="store_true", default=False)
    parser.add_argument("--cond_mask_prob", type=float, default=0.0)
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--sample_period", type=int, default=-1)
    parser.add_argument("--state_noise_snr", type=float, default=None)
    parser.add_argument("--auto_adjust_image_brightness", action="store_true", default=False)
    parser.add_argument("--precomp_lang_embed", action="store_true", default=False)
    parser.add_argument("--cam_ext_mask_prob", type=float, default=-1.0)
    parser.add_argument("--enable_distill", action="store_true", default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
