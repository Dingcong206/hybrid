#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
import tensorflow as tf
from tqdm import tqdm

# =========================
# Constants
# =========================
SR = 16000
DURATION_SEC = 2
TARGET_SAMPLES = SR * DURATION_SEC  # 32000

# =========================
# Label mapping (ICBHI 4-class)
# =========================
def lungsound_label(crackles: int, wheezes: int) -> int:
    c, w = int(crackles), int(wheezes)
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3

# =========================
# Audio loading
# =========================
def load_wav_resample_mono(wav_path: str, target_sr: int = SR) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(wav_path)  # (C, N)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = T.Resample(orig_sr, target_sr)(wav)
    return wav  # (1, N)

# =========================
# Make 2s: energy-max crop if long; center-pad if short
# =========================
def process_to_2s_sample(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: (1, N)
    return: (1, 32000)
    """
    n = wav.shape[-1]
    if n > TARGET_SAMPLES:
        # sliding window find max energy, step=100ms
        step = 1600  # 0.1s
        windows = wav.unfold(-1, TARGET_SAMPLES, step)  # (1, num_win, 32000)
        energies = torch.sum(windows ** 2, dim=-1)      # (1, num_win)
        best_idx = torch.argmax(energies, dim=-1).item()
        return windows[0, best_idx].unsqueeze(0)        # (1,32000)
    elif n < TARGET_SAMPLES:
        pad_total = TARGET_SAMPLES - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(wav, (pad_left, pad_right))
    else:
        return wav

# =========================
# TF SavedModel helpers
# =========================
def load_tf_model(model_dir: str):
    m = tf.saved_model.load(model_dir)
    sigs = list(m.signatures.keys())
    if "serving_default" not in m.signatures:
        raise RuntimeError(f"No serving_default in signatures: {sigs}")
    fn = m.signatures["serving_default"]
    # auto detect input key
    in_keys = list(fn.structured_input_signature[1].keys())
    if len(in_keys) == 0:
        raise RuntimeError("No input keys found in model signature.")
    feed_key = in_keys[0]
    return m, fn, feed_key, sigs

def tf_forward_frontend(fn, feed_key: str, wav_2s: torch.Tensor) -> np.ndarray:
    """
    Input: wav_2s torch (1, 32000)
    Output: feats numpy (T, 48)  [expected from spectrogram_frontend]
    """
    audio_tf = tf.constant(wav_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)
    out = fn(**{feed_key: audio_tf})
    # take first output
    first = next(iter(out.values()))
    arr = first.numpy().squeeze(0)  # (T, 48) typically
    return arr

# =========================
# Token projection: (T,48) -> (T,1024)
# =========================
def project_48_to_1024(x48: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    x48: (T,48)
    return: (T,1024)
    Note: This is a learnable projection in a real model.
          Here we initialize a fixed random matrix so you can SAVE tokens offline.
          Better: do this projection inside your PyTorch model and train it.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((48, 1024)).astype(np.float32) / math.sqrt(48.0)
    b = np.zeros((1024,), dtype=np.float32)
    x = x48.astype(np.float32) @ W + b
    return x

def downsample_to_128(x: np.ndarray) -> np.ndarray:
    """
    x: (T,1024) -> (128,1024) using linear interpolation along time
    """
    xt = torch.from_numpy(x).transpose(0, 1).unsqueeze(0)  # (1,1024,T)
    yt = F.interpolate(xt, size=128, mode="linear", align_corners=False)
    y = yt.squeeze(0).transpose(0, 1).cpu().numpy()        # (128,1024)
    return y

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/dingcong/hybrid/audio_and_txt_files")
    parser.add_argument("--out_dir", type=str, default="/data/dingcong/hybrid/icbhi_tokens_T_1024")
    parser.add_argument("--frontend_path", type=str, required=True,
                        help="TF SavedModel dir for spectrogram_frontend")
    parser.add_argument("--split_file", type=str, default="/data/dingcong/hybrid/audio_and_txt_files/official_split.txt")
    parser.add_argument("--force_128", action="store_true",
                        help="if set, downsample tokens to (128,1024) before saving")
    parser.add_argument("--proj_seed", type=int, default=42,
                        help="random seed for offline projection 48->1024 (better to learn in-model)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"📦 Loading spectrogram_frontend from: {args.frontend_path}")
    _, fn, feed_key, sigs = load_tf_model(args.frontend_path)
    print(f"✅ signatures: {sigs}")
    print(f"✅ input key: {feed_key}")

    split_df = pd.read_csv(args.split_file, sep="\t", names=["file", "set"])
    split_map = dict(zip(split_df["file"], split_df["set"]))

    wav_files = list(Path(args.data_dir).glob("*.wav"))
    print(f"🔍 Found {len(wav_files)} wav files.")

    index_rows = []

    for wav_path in tqdm(wav_files):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        tag = split_map.get(wav_path.stem) or split_map.get(wav_path.name)
        if not tag:
            continue

        # load full recording
        wav = load_wav_resample_mono(str(wav_path))  # (1,N)

        # read cycles
        ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])
        patient_id = wav_path.stem.split("_")[0]

        for i, r in ann.iterrows():
            start_i = int(float(r["Start"]) * SR)
            end_i   = int(float(r["End"])   * SR)
            cycle = wav[:, start_i:end_i]
            if cycle.shape[-1] < 1600:
                continue

            y = lungsound_label(r["C"], r["W"])
            cycle_2s = process_to_2s_sample(cycle)  # (1,32000)

            # 1) frontend feats (T,48)
            x48 = tf_forward_frontend(fn, feed_key, cycle_2s)

            # 2) project to tokens (T,1024)
            x1024 = project_48_to_1024(x48, seed=args.proj_seed)

            # 3) optional: force to (128,1024)
            tokens = downsample_to_128(x1024) if args.force_128 else x1024

            # save
            save_subdir = Path(args.out_dir) / tag / patient_id
            save_subdir.mkdir(parents=True, exist_ok=True)

            npy_name = f"{wav_path.stem}_cycle{i}_y{y}.npy"
            npy_path = save_subdir / npy_name
            np.save(npy_path, tokens)

            index_rows.append({
                "tokens_path": str(npy_path),
                "label": y,
                "recording": wav_path.stem,
                "patient_id": patient_id,
                "set": tag,
                "shape": str(tokens.shape),
            })

    df = pd.DataFrame(index_rows)
    df[df["set"] == "train"].to_csv(Path(args.out_dir) / "train_index.csv", index=False)
    df[df["set"] == "test"].to_csv(Path(args.out_dir) / "test_index.csv", index=False)

    print("\n✨ Done.")
    if len(df) > 0:
        print("Example saved shape:", df.iloc[-1]["shape"])
    print("train:", (df["set"] == "train").sum(), " test:", (df["set"] == "test").sum())

if __name__ == "__main__":
    main()
