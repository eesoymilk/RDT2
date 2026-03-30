#!/usr/bin/env python3
"""Convert a UMI zarr replay buffer to RDT-2 FM WebDataset format (single-arm).

Each output sample is one action chunk:
  {idx}.image.jpg   — (384, 768, 3) uint8: single wrist cam duplicated side-by-side
  {idx}.action.npy  — (24, 20) float32: right-arm 10-D + zero-padded left-arm 10-D
  {idx}.meta.json   — {"sub_task_instruction_key": ..., "task_id": ..., "user_id": ...}

Action layout per arm (10-D):
  dims 0-2   : EEF position relative to chunk-start frame (metres)
  dims 3-8   : EEF rotation in 6-D continuous rep (first two columns of rot matrix)
  dim  9     : gripper width (raw metres, normaliser handles scaling)
Left arm (dims 10-19) is zero-padded for single-arm robots.

Usage:
  uv run --with "zarr<3,imagecodecs,webdataset,pillow,scipy" \\
      scripts/zarr_to_webdataset.py \\
      --zarr_path  umi_data/mug_handover_general/replay_buffer \\
      --out_dir    data/mug_handover_webdataset \\
      --task_key   mug_handover \\
      --instruction "Pick up the mug and hand it to the user."
"""

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.umi.codecs.imagecodecs_numcodecs import register_codecs

import zarr

# ── constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 24      # action horizon expected by RDT-2 FM
IMAGE_OUT = 384      # each eye is (384, 384); tiled image is (384, 768)
JPEG_QUALITY = 95


# ── helpers ──────────────────────────────────────────────────────────────────

def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(T, 3) axis-angle → (T, 3, 3) rotation matrices."""
    return Rotation.from_rotvec(aa).as_matrix()


def build_pose_mats(pos: np.ndarray, rot_aa: np.ndarray) -> np.ndarray:
    """Build (T, 4, 4) homogeneous pose matrices from pos (T,3) and rot_aa (T,3)."""
    T = len(pos)
    mats = np.zeros((T, 4, 4), dtype=np.float64)
    mats[:, :3, :3] = axis_angle_to_rotmat(rot_aa)
    mats[:, :3, 3] = pos
    mats[:, 3, 3] = 1.0
    return mats


def mat_to_pose9d(mat: np.ndarray) -> np.ndarray:
    """(T, 4, 4) → (T, 9): [pos(3), rot_col0(3), rot_col1(3)] (no gripper)."""
    pos = mat[:, :3, 3]                        # (T, 3)
    col0 = mat[:, :3, 0]                        # (T, 3)
    col1 = mat[:, :3, 1]                        # (T, 3)
    return np.concatenate([pos, col0, col1], axis=-1)  # (T, 9)


def build_action_chunk(
    pos: np.ndarray,       # (T_ep, 3)
    rot_aa: np.ndarray,    # (T_ep, 3)
    gripper: np.ndarray,   # (T_ep, 1)
    start: int,
) -> np.ndarray:
    """Return (CHUNK_SIZE, 20) float32 action chunk relative to frame `start`.

    Frames past the episode end are padded by repeating the last frame.
    Right arm in dims 0-9; left arm in dims 10-19 (zero-padded).
    """
    T_ep = len(pos)
    end = min(start + CHUNK_SIZE, T_ep)
    pad = CHUNK_SIZE - (end - start)

    chunk_pos = pos[start:end]
    chunk_rot = rot_aa[start:end]
    chunk_grip = gripper[start:end]

    if pad > 0:
        chunk_pos = np.concatenate([chunk_pos, np.tile(chunk_pos[-1:], (pad, 1))])
        chunk_rot = np.concatenate([chunk_rot, np.tile(chunk_rot[-1:], (pad, 1))])
        chunk_grip = np.concatenate([chunk_grip, np.tile(chunk_grip[-1:], (pad, 1))])

    pose_mats = build_pose_mats(chunk_pos, chunk_rot)          # (24, 4, 4)
    base_inv = np.linalg.inv(pose_mats[0])                     # (4, 4)
    rel_mats = base_inv[None] @ pose_mats                       # (24, 4, 4)

    pose9d = mat_to_pose9d(rel_mats).astype(np.float32)        # (24, 9)
    right_arm = np.concatenate([pose9d, chunk_grip.astype(np.float32)], axis=-1)  # (24, 10)
    left_arm = np.zeros_like(right_arm)                         # (24, 10)  — zero-pad
    return np.concatenate([right_arm, left_arm], axis=-1)       # (24, 20)


def encode_image(frame: np.ndarray) -> bytes:
    """Resize (H, W, 3) uint8 → (IMAGE_OUT, IMAGE_OUT), tile side-by-side, JPEG-encode."""
    img = Image.fromarray(frame).resize((IMAGE_OUT, IMAGE_OUT), Image.LANCZOS)
    arr = np.asarray(img)
    tiled = np.concatenate([arr, arr], axis=1)   # (384, 768, 3)
    buf = io.BytesIO()
    Image.fromarray(tiled).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def add_to_tar(tar: tarfile.TarFile, key: str, ext: str, data: bytes) -> None:
    name = f"{key}.{ext}"
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ── main ─────────────────────────────────────────────────────────────────────

def convert(
    zarr_path: str,
    out_dir: str,
    task_key: str,
    instruction: str,
    stride: int,
    shard_size: int,
    skip_short: bool,
) -> None:
    register_codecs()
    os.makedirs(out_dir, exist_ok=True)

    # Write instructions file
    with open(os.path.join(out_dir, "instructions.json"), "w") as f:
        json.dump({task_key: instruction}, f, indent=2)
    print(f"Wrote instructions.json  ({task_key!r}: {instruction!r})")

    root = zarr.open(zarr.ZipStore(zarr_path, mode="r"))
    episode_ends = root["meta"]["episode_ends"][:]   # (E,) int64

    n_eps = len(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])

    sample_idx = 0
    shard_idx = 0
    n_skipped = 0
    tar: tarfile.TarFile | None = None

    def open_shard() -> tarfile.TarFile:
        p = os.path.join(out_dir, f"shard-{shard_idx:06d}.tar")
        print(f"  Opening shard {p}")
        return tarfile.open(p, "w")

    tar = open_shard()

    for ep_idx in range(n_eps):
        ep_s = int(ep_starts[ep_idx])
        ep_e = int(episode_ends[ep_idx])
        ep_len = ep_e - ep_s

        if ep_len < CHUNK_SIZE:
            if skip_short:
                print(f"  Episode {ep_idx}: length {ep_len} < {CHUNK_SIZE}, skipping.")
                n_skipped += 1
                continue
            # else we still write the one (padded) chunk starting at frame 0

        # Load this episode's data lazily (avoids holding all images in RAM at once)
        ep_rgb = root["data"]["camera0_rgb"][ep_s:ep_e]              # (L, 224, 224, 3)
        ep_pos = root["data"]["robot0_eef_pos"][ep_s:ep_e]           # (L, 3)
        ep_rot = root["data"]["robot0_eef_rot_axis_angle"][ep_s:ep_e] # (L, 3)
        ep_grip = root["data"]["robot0_gripper_width"][ep_s:ep_e]    # (L, 1)
        task_id = int(root["data"]["task_id"][ep_s])
        user_id = int(root["data"]["user_id"][ep_s])

        for cs in range(0, ep_len, stride):
            # Roll over shard
            if sample_idx > 0 and sample_idx % shard_size == 0:
                tar.close()
                shard_idx += 1
                tar = open_shard()

            key = f"{sample_idx:08d}"

            img_bytes = encode_image(ep_rgb[cs])

            action = build_action_chunk(ep_pos, ep_rot, ep_grip, cs)
            action_buf = io.BytesIO()
            np.save(action_buf, action)

            meta = {
                "sub_task_instruction_key": task_key,
                "task_id": task_id,
                "user_id": user_id,
                "episode_idx": ep_idx,
                "chunk_start_frame": ep_s + cs,
            }

            add_to_tar(tar, key, "image.jpg", img_bytes)
            add_to_tar(tar, key, "action.npy", action_buf.getvalue())
            add_to_tar(tar, key, "meta.json", json.dumps(meta).encode())

            sample_idx += 1

        if (ep_idx + 1) % 10 == 0 or ep_idx == n_eps - 1:
            print(f"  Episodes processed: {ep_idx + 1}/{n_eps}  samples so far: {sample_idx}")

    if tar is not None:
        tar.close()

    print(f"\nDone.")
    print(f"  Episodes:       {n_eps - n_skipped} written, {n_skipped} skipped (too short)")
    print(f"  Total samples:  {sample_idx}")
    print(f"  Shards:         {shard_idx + 1}")
    print(f"  Output dir:     {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UMI zarr replay buffer to RDT-2 FM WebDataset (single-arm)"
    )
    parser.add_argument("--zarr_path", required=True,
                        help="Path to zarr ZipStore replay_buffer file")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for WebDataset shards")
    parser.add_argument("--task_key", default="mug_handover",
                        help="Key used in instructions.json (default: mug_handover)")
    parser.add_argument("--instruction",
                        default="Pick up the mug and hand it to the user.",
                        help="Natural language instruction for this task")
    parser.add_argument("--stride", type=int, default=1,
                        help="Chunk stride: 1=overlapping (default), 24=non-overlapping")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of samples per tar shard (default: 1000)")
    parser.add_argument("--skip_short", action="store_true",
                        help="Skip episodes shorter than CHUNK_SIZE instead of padding them")
    args = parser.parse_args()

    convert(
        zarr_path=args.zarr_path,
        out_dir=args.out_dir,
        task_key=args.task_key,
        instruction=args.instruction,
        stride=args.stride,
        shard_size=args.shard_size,
        skip_short=args.skip_short,
    )


if __name__ == "__main__":
    main()
