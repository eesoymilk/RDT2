#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/ucvla/stage1"

# Find latest checkpoint
checkpoints = sorted(Path(ckpt_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if not checkpoints:
    print(f"No checkpoints found in {ckpt_dir}")
    sys.exit(1)

ckpt_path = checkpoints[-1] / "ucvla_weights.pt"
print(f"Checking: {ckpt_path}\n")

ckpt = torch.load(ckpt_path, map_location="cpu")
w = ckpt["user_bias"]["weight"]  # (n_users, d_bias)
n_users = w.shape[0]

print("── Bias norms ──────────────────────────────")
for i in range(n_users):
    print(f"  user_{i}: {w[i].norm().item():.6f}")

print("\n── Cosine similarities (should diverge → 0) ─")
for i in range(n_users):
    for j in range(i + 1, n_users):
        cos = F.cosine_similarity(w[i], w[j], dim=0).item()
        print(f"  cos(user_{i}, user_{j}): {cos:.6f}")

print("\n── bias_proj weight norm ───────────────────")
bp = ckpt["bias_proj"]["weight"]
print(f"  bias_proj.weight norm: {bp.norm().item():.6f}")
