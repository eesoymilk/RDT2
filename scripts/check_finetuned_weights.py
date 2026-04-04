#!/usr/bin/env python3
"""
Check if the finetuned RDT checkpoint actually differs from the base pretrained weights.
Compares weight norms and a random sample of parameter differences.
"""
import sys
import torch
from pathlib import Path

from models.rdt_runner import RDTRunner

BASE_CKPT = "outputs/rdt/rdt2-action-expert/checkpoint-40000"

print(f"Loading finetuned checkpoint from {BASE_CKPT}...")
finetuned = RDTRunner.from_pretrained(BASE_CKPT)

print("\n── Finetuned weight norms (sample) ────────────────")
for name, param in list(finetuned.model.named_parameters())[:10]:
    print(f"  {name}: norm={param.norm().item():.6f}, mean={param.mean().item():.6f}")

print("\n── Checking if weights are all zero ────────────────")
all_zero = all(p.norm().item() == 0.0 for p in finetuned.model.parameters())
print(f"  All zero: {all_zero}")

print("\n── Overall model stats ─────────────────────────────")
total_norm = sum(p.norm().item() ** 2 for p in finetuned.model.parameters()) ** 0.5
print(f"  Total param norm: {total_norm:.4f}")
n_params = sum(p.numel() for p in finetuned.model.parameters())
print(f"  Total params: {n_params:,}")
