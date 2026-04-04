#!/usr/bin/env python3
"""
Compare two RDT checkpoints to verify weights actually changed during finetuning.
"""
import torch
from pathlib import Path
from models.rdt_runner import RDTRunner

CKPT_A = "outputs/rdt/rdt2-action-expert/checkpoint-35000"
CKPT_B = "outputs/rdt/rdt2-action-expert/checkpoint-40000"

print(f"Loading {CKPT_A}...")
model_a = RDTRunner.from_pretrained(CKPT_A)
print(f"Loading {CKPT_B}...")
model_b = RDTRunner.from_pretrained(CKPT_B)

params_a = dict(model_a.model.named_parameters())
params_b = dict(model_b.model.named_parameters())

print("\n── Per-layer diff norms (sample) ───────────────────")
total_diff = 0.0
n_changed = 0
for name in list(params_a.keys())[:15]:
    diff = (params_a[name] - params_b[name]).norm().item()
    total_diff += diff
    if diff > 0:
        n_changed += 1
    print(f"  {name}: diff_norm={diff:.6f}")

print(f"\n── Summary ─────────────────────────────────────────")
all_diffs = [(params_a[n] - params_b[n]).norm().item() for n in params_a]
changed = sum(1 for d in all_diffs if d > 0)
print(f"  Layers with any change: {changed} / {len(all_diffs)}")
print(f"  Max diff norm: {max(all_diffs):.6f}")
print(f"  Mean diff norm: {sum(all_diffs)/len(all_diffs):.6f}")
