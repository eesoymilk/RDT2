#!/usr/bin/env python3
"""
Cross-user confusion matrix for UCVLA Stage 1.
For each val sample from user u, predict with all user biases.
The correct bias should produce lower action MSE.
"""
import argparse
import json
import socket
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.normalizer import LinearNormalizer
from models.rdt_runner import RDTRunner
from models.ucvla.ucvla_runner import UCVLARDTRunner
from rdt.dataset import get_val_dataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/rdt/post_train.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="outputs/rdt/rdt2-action-expert/checkpoint-40000")
    parser.add_argument("--pretrained_vision_language_model_name_or_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--ucvla_weights", type=str,
                        default="outputs/ucvla/stage1/ucvla_weights.pt")
    parser.add_argument("--webdataset_config", type=str,
                        default="configs/datasets/mug_handover.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left", use_fast=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        torch_dtype=dtype, attn_implementation="flash_attention_2", device_map=device)
    vlm.eval()

    print("Loading RDT...")
    base_rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)

    print(f"Loading UCVLA weights from {args.ucvla_weights}...")
    ckpt = torch.load(args.ucvla_weights, map_location="cpu")
    n_users = ckpt["n_users"]
    d_bias = ckpt["d_bias"]

    ucvla = UCVLARDTRunner(base_rdt, n_users=n_users, d_bias=d_bias)
    ucvla.ucvla_model.user_bias.load_state_dict(ckpt["user_bias"])
    ucvla.ucvla_model.bias_proj.load_state_dict(ckpt["bias_proj"])
    ucvla.eval().to(device)

    with open(args.webdataset_config) as f:
        hostname = socket.gethostname()
        wds_config = yaml.safe_load(f.read().format(hostname=hostname))

    with open(wds_config["kwargs"]["instruction_path"]) as f:
        instructions = json.load(f)

    val_dataset = get_val_dataset(wds_config["shards_dir"])
    val_collate = partial(collate_fn, processor=processor, instructions=instructions,
                          image_corruption=False, state_dim=config["common"]["state_dim"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=val_collate, num_workers=2)

    normalizer = LinearNormalizer.load(wds_config["kwargs"]["normalizer_path"])

    errors = defaultdict(list)  # (uid_true, uid_pred) -> [mse, ...]
    selected_layers = config["model"]["selected_layers"]

    print(f"\nRunning cross-user eval (n_users={n_users})...")
    with torch.no_grad():
        for batch in val_loader:
            actions = batch["actions"]
            nsamples = normalizer["action"].normalize(actions).to(dtype=dtype, device=device)
            states = batch["states"].to(dtype=dtype, device=device)
            uid_true = batch["user_id"].to(device)

            with torch.autocast("cuda", dtype=dtype):
                outputs = vlm(**batch["vision_language_model_inputs"].to(device), use_cache=True)
            if isinstance(selected_layers, list):
                kv = [outputs.past_key_values[i] for i in selected_layers]
            else:
                kv = [outputs.past_key_values[selected_layers]]
            lang_mask = batch["vision_language_model_inputs"]["attention_mask"].to(
                dtype=torch.bool, device=device)

            for uid_pred in range(n_users):
                uid_tensor = torch.full_like(uid_true, uid_pred)
                with torch.autocast("cuda", dtype=dtype):
                    pred = ucvla.predict_action(
                        user_id=uid_tensor, state_tokens=states,
                        lang_kv_cache=kv, lang_attn_mask=lang_mask)
                err = F.mse_loss(pred, nsamples, reduction="none").mean(dim=[1, 2])
                for i, ut in enumerate(uid_true.tolist()):
                    errors[(ut, uid_pred)].append(err[i].item())

    print("\n── Cross-user confusion matrix (MSE, lower=better) ────────────")
    print("   rows=true user, cols=predicted bias\n")
    header = "        " + "  ".join(f"bias_{j}" for j in range(n_users))
    print(header)
    for ut in range(n_users):
        row = []
        for up in range(n_users):
            vals = errors[(ut, up)]
            mean = sum(vals) / len(vals) if vals else float("nan")
            marker = " ←" if up == ut else ""
            row.append(f"{mean:.5f}{marker}")
        print(f"  user_{ut}: " + "  ".join(row))

    print("\n── Diagonal dominance check ────────────────────────────────────")
    wins = 0
    for ut in range(n_users):
        correct = sum(errors[(ut, ut)]) / len(errors[(ut, ut)])
        others = [sum(errors[(ut, up)]) / len(errors[(ut, up)])
                  for up in range(n_users) if up != ut]
        if correct < min(others):
            wins += 1
            print(f"  user_{ut}: correct bias WINS ✓  ({correct:.5f} < {min(others):.5f})")
        else:
            print(f"  user_{ut}: correct bias LOSES ✗  ({correct:.5f} vs min_other={min(others):.5f})")
    print(f"\n  {wins}/{n_users} users have correct bias winning")


if __name__ == "__main__":
    main()
