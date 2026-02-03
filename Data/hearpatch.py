#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import tensorflow as tf
from tqdm import tqdm

# ============================
# 1. 常量与基础配置
# ============================
SR = 16000
TARGET_SAMPLES = 32000


def lungsound_label(crackles, wheezes):
    c, w = int(crackles), int(wheezes)
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3


def process_to_2s_sample(wav: torch.Tensor) -> torch.Tensor:
    if wav.shape[-1] > TARGET_SAMPLES:
        step = 1600
        windows = wav.unfold(-1, TARGET_SAMPLES, step)
        energies = torch.sum(windows ** 2, dim=-1)
        best_idx = torch.argmax(energies)
        return windows[0, best_idx].unsqueeze(0)
    else:
        pad_total = TARGET_SAMPLES - wav.shape[-1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.nn.functional.pad(wav, (pad_left, pad_right))


# ============================
# 2. 核心处理逻辑
# ============================
def process_subset(subset_dir, save_subset_dir, extract_fn):
    wav_files = list(Path(subset_dir).glob("*.wav"))
    if not wav_files:
        return []

    print(f"📂 正在从 {subset_dir} 提取特征...")
    rows = []
    for wav_path in tqdm(wav_files):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists(): continue

        wav, orig_sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != SR: wav = torchaudio.transforms.Resample(orig_sr, SR)(wav)

        ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])
        patient_id = wav_path.stem.split("_")[0]

        for i, r in ann.iterrows():
            start_i, end_i = int(r["Start"] * SR), int(r["End"] * SR)
            cycle = wav[:, start_i:end_i]
            if cycle.shape[-1] < 1600: continue

            cycle_2s = process_to_2s_sample(cycle)
            label = lungsound_label(r["C"], r["W"])

            # HeAR 推理 [128, 1024]
            audio_tf = tf.constant(cycle_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)
            outputs = extract_fn(audio_wav=audio_tf)
            tokens = list(outputs.values())[0].numpy().squeeze(0)

            # 强制创建子目录
            save_path = Path(save_subset_dir) / patient_id
            save_path.mkdir(parents=True, exist_ok=True)

            npy_name = f"{wav_path.stem}_c{i}_y{label}.npy"
            npy_full_path = save_path / npy_name
            np.save(npy_full_path, tokens)

            rows.append({
                "tokens_path": str(npy_full_path),
                "label": label,
                "recording": wav_path.stem,
                "patient_id": patient_id
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    # 路径已根据你的截图修正为 ast
    parser.add_argument("--src_root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--save_root", type=str, default="/data/dingcong/hybrid/icbhi_hear_patch_128_1024")
    parser.add_argument("--hear_path", type=str,
                        default="/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f/event_detector/spectrogram_frontend")
    args = parser.parse_args()

    # --- 修正点：首先创建根目录 ---
    os.makedirs(args.save_root, exist_ok=True)

    print("📦 Loading HeAR Model...")
    model = tf.saved_model.load(args.hear_path)
    extract_fn = model.signatures["serving_default"]

    for subset in ["train", "test"]:
        src_dir = Path(args.src_root) / subset
        dst_dir = Path(args.save_root) / subset

        if src_dir.exists():
            indices = process_subset(src_dir, dst_dir, extract_fn)
            if indices:
                csv_path = Path(args.save_root) / f"{subset}_index.csv"
                pd.DataFrame(indices).to_csv(csv_path, index=False)
                print(f"✅ {subset} 索引已保存至: {csv_path}")
        else:
            print(f"⚠️  警告：源目录不存在 {src_dir}")

    print(f"\n🚀 所有流程已完成！")


if __name__ == "__main__":
    main()