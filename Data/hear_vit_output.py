#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

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


def lungsound_label(crackles: int, wheezes: int) -> int:
    c, w = int(crackles), int(wheezes)
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3


def load_wav_resample_mono(wav_path: str, target_sr: int = SR) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(wav_path)  # (C, N)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = T.Resample(orig_sr, target_sr)(wav)
    return wav  # (1, N)


def process_to_2s_sample(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: (1, N)
    return: (1, 32000)
    """
    n = wav.shape[-1]
    if n > TARGET_SAMPLES:
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


def load_tf_model(model_dir: str):
    m = tf.saved_model.load(model_dir)
    sigs = list(m.signatures.keys())
    if "serving_default" not in m.signatures:
        raise RuntimeError(f"No serving_default in signatures: {sigs}")
    fn = m.signatures["serving_default"]

    in_keys = list(fn.structured_input_signature[1].keys())
    if len(in_keys) == 0:
        raise RuntimeError("No input keys found in model signature.")
    feed_key = in_keys[0]
    return m, fn, feed_key, sigs


def pick_vit_output(outputs: dict):
    """
    从 SavedModel 输出里“智能选择” ViT 之后的输出。
    优先顺序：
      embedding / embeddings / representation / representations / features / pooled
    否则取第一个输出。
    """
    keys = list(outputs.keys())
    lower_map = {k: k.lower() for k in keys}

    prefer = [
        "embedding", "embeddings",
        "representation", "representations",
        "feature", "features",
        "pooled", "pooler", "cls",
        "vit", "transformer"
    ]

    for p in prefer:
        for k in keys:
            if p in lower_map[k]:
                return k, outputs[k]

    # fallback
    k0 = keys[0]
    return k0, outputs[k0]


def tf_forward_vit(fn, feed_key: str, wav_2s: torch.Tensor) -> np.ndarray:
    """
    输入: wav_2s torch (1, 32000)
    输出: numpy array (可能是 (D,) 或 (T,D) 或 (1,T,D) 去掉 batch 后)
    """
    audio_tf = tf.constant(wav_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)
    out = fn(**{feed_key: audio_tf})

    out_key, out_tensor = pick_vit_output(out)
    arr = out_tensor.numpy()

    # 去掉 batch 维（如果有）
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    return out_key, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/dingcong/hybrid/audio_and_txt_files")
    parser.add_argument("--out_dir", type=str, default="/data/dingcong/hybrid/icbhi_hear_vit_outputs")
    parser.add_argument("--hear_model_path", type=str, required=True,
                        help="TF SavedModel dir for event_detector_large (包含ViT)")
    parser.add_argument("--split_file", type=str, default="/data/dingcong/hybrid/audio_and_txt_files/official_split.txt")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"📦 Loading HeAR model (with ViT) from: {args.hear_model_path}")
    _, fn, feed_key, sigs = load_tf_model(args.hear_model_path)
    print(f"✅ signatures: {sigs}")
    print(f"✅ input key: {feed_key}")

    split_df = pd.read_csv(args.split_file, sep="\t", names=["file", "set"])
    split_map = dict(zip(split_df["file"], split_df["set"]))

    wav_files = list(Path(args.data_dir).glob("*.wav"))
    print(f"🔍 Found {len(wav_files)} wav files.")

    index_rows = []
    example_shape = None
    example_key = None

    for wav_path in tqdm(wav_files):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        tag = split_map.get(wav_path.stem) or split_map.get(wav_path.name)
        if not tag:
            continue

        wav = load_wav_resample_mono(str(wav_path))  # (1,N)
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

            out_key, vit_out = tf_forward_vit(fn, feed_key, cycle_2s)

            # 保存
            save_subdir = Path(args.out_dir) / tag / patient_id
            save_subdir.mkdir(parents=True, exist_ok=True)

            npy_name = f"{wav_path.stem}_cycle{i}_y{y}.npy"
            npy_path = save_subdir / npy_name
            np.save(npy_path, vit_out)

            if example_shape is None:
                example_shape = vit_out.shape
                example_key = out_key

            index_rows.append({
                "tokens_path": str(npy_path),
                "label": y,
                "recording": wav_path.stem,
                "patient_id": patient_id,
                "set": tag,
                "out_key": out_key,
                "shape": str(vit_out.shape),
            })

    df = pd.DataFrame(index_rows)
    df[df["set"] == "train"].to_csv(Path(args.out_dir) / "train_index.csv", index=False)
    df[df["set"] == "test"].to_csv(Path(args.out_dir) / "test_index.csv", index=False)

    print("\n✨ Done.")
    print("Example output key:", example_key)
    print("Example output shape:", example_shape)
    print("train:", (df["set"] == "train").sum(), " test:", (df["set"] == "test").sum())


if __name__ == "__main__":
    main()
