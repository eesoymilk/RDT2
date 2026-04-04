#!/usr/bin/env python3
"""
Reshard mug_handover_webdataset into balanced train/val splits.
Holds out VAL_PER_USER samples per user for validation (all 3 users in val).
Rewrites train shards + one val shard in-place (originals backed up).
"""
import io
import json
import os
import random
import tarfile
from collections import defaultdict, Counter
from pathlib import Path

SHARDS_DIR = Path("data/mug_handover_webdataset")
VAL_PER_USER = 100
TRAIN_SHARDS = 3
SEED = 42

random.seed(SEED)

# ── 1. Read all samples from all shards ──────────────────────────────────────
print("Reading all shards...")
user_samples = defaultdict(list)

for shard_path in sorted(SHARDS_DIR.glob("shard-*.tar")):
    print(f"  {shard_path.name}")
    with tarfile.open(shard_path) as tf:
        by_index = defaultdict(dict)
        for m in tf.getmembers():
            idx, ext = m.name.split(".", 1)
            by_index[idx][ext] = tf.extractfile(m).read()
        for idx, files in by_index.items():
            meta = json.loads(files["meta.json"])
            user_id = meta.get("user_id", -1)
            user_samples[user_id].append(files)

for uid, samples in sorted(user_samples.items()):
    print(f"  user {uid}: {len(samples)} samples")

# ── 2. Split per user ────────────────────────────────────────────────────────
train_samples = []
val_samples = []

for uid in sorted(user_samples.keys()):
    samples = user_samples[uid].copy()
    random.shuffle(samples)
    val_samples.extend(samples[:VAL_PER_USER])
    train_samples.extend(samples[VAL_PER_USER:])

random.shuffle(train_samples)
random.shuffle(val_samples)

print(f"\nTrain: {len(train_samples)} samples")
print(f"Val:   {len(val_samples)} samples")

# ── 3. Backup originals ──────────────────────────────────────────────────────
backup_dir = SHARDS_DIR / "backup_original_shards"
backup_dir.mkdir(exist_ok=True)
for shard_path in sorted(SHARDS_DIR.glob("shard-*.tar")):
    dest = backup_dir / shard_path.name
    if not dest.exists():
        print(f"Backing up {shard_path.name}...")
        shard_path.rename(dest)
    else:
        shard_path.unlink()
        print(f"Removed {shard_path.name} (backup already exists)")

# ── 4. Write new shards ──────────────────────────────────────────────────────
def write_shard(path: Path, samples: list, start_idx: int):
    with tarfile.open(path, "w") as tf:
        for i, files in enumerate(samples):
            key = f"{start_idx + i:06d}"
            for ext, data in files.items():
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    print(f"  Wrote {path.name} ({len(samples)} samples)")

print(f"\nWriting {TRAIN_SHARDS} train shards...")
chunk = len(train_samples) // TRAIN_SHARDS
idx = 0
for s in range(TRAIN_SHARDS):
    start = s * chunk
    end = start + chunk if s < TRAIN_SHARDS - 1 else len(train_samples)
    write_shard(SHARDS_DIR / f"shard-{s:06d}.tar", train_samples[start:end], idx)
    idx += end - start

print(f"\nWriting val shard...")
write_shard(SHARDS_DIR / f"shard-{TRAIN_SHARDS:06d}.tar", val_samples, idx)

# ── 5. Verify ────────────────────────────────────────────────────────────────
print("\nVerifying splits:")
for shard_path in sorted(SHARDS_DIR.glob("shard-*.tar")):
    counts = Counter()
    with tarfile.open(shard_path) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".meta.json"):
                meta = json.loads(tf.extractfile(m).read())
                counts[meta.get("user_id", -1)] += 1
    print(f"  {shard_path.name}: {dict(sorted(counts.items()))}")

print(f"\nDone. Originals backed up to: {backup_dir}")
