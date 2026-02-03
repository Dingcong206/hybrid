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
# 1. 常量设置
# ============================
SR = 16000
TARGET_SAMPLES = 32000  # 2秒音频


def lungsound_label(crackles, wheezes):
    c, w = int(crackles), int(wheezes)
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3


# ============================
# 2. 核心 2s 裁剪/填充逻辑
# ============================
def process_to_2s_sample(wav: torch.Tensor) -> torch.Tensor:
    """根据能量截取 2s 或填充至 2s"""
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
# 3. 主处理函数
# ============================
def process_subset(subset_dir, save_subset_dir, extract_fn, device_tag):
    """
    处理特定的子集 (train 或 test)
    """
    wav_files = list(Path(subset_dir).glob("*.wav"))
    print(f"📂 Processing {len(wav_files)} files in {subset_dir}...")

    rows = []
    for wav_path in tqdm(wav_files):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists(): continue

        # 加载音频并重采样
        wav, orig_sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != SR: wav = torchaudio.transforms.Resample(orig_sr, SR)(wav)

        # 解析周期标注
        ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])
        patient_id = wav_path.stem.split("_")[0]

        for i, r in ann.iterrows():
            start_i, end_i = int(r["Start"] * SR), int(r["End"] * SR)
            cycle = wav[:, start_i:end_i]
            if cycle.shape[-1] < 1600: continue  # 过滤噪声短音频

            # 智能裁剪/填充到 2s
            cycle_2s = process_to_2s_sample(cycle)
            label = lungsound_label(r["C"], r["W"])

            # --- HeAR 提取 [128, 1024] Tokens ---
            audio_tf = tf.constant(cycle_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)
            # 调用 HeAR 模型接口
            outputs = extract_fn(audio_wav=audio_tf)
            # 获取 Patch Embeddings (通常是输出字典中的第一个值)
            # 形状应为 [1, 128, 1024] -> squeeze 得到 [128, 1024]
            tokens = list(outputs.values())[0].numpy().squeeze(0)

            # 保存结果，按患者 ID 分类
            save_path = Path(save_subset_dir) / patient_id
            save_path.mkdir(parents=True, exist_ok=True)
            npy_name = f"{wav_path.stem}_c{i}_y{label}.npy"
            npy_full_path = save_path / npy_name
            np.save(npy_full_path, tokens)

            rows.append({
                "tokens_path": str(npy_full_path),
                "label": label,
                "recording": wav_path.stem
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    # 输入路径指向你已经划分好的父目录
    parser.add_argument("--src_root", type=str, default="/data/dingcong/hybrid/icbhi_official_sat_patch_tokens")
    parser.add_argument("--save_root", type=str, default="/data/dingcong/hybrid/icbhi_hear_patch_128_1024")
    parser.add_argument("--hear_path", type=str, required=True, help="/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f/event_detector/spectrogram_frontend")
    args = parser.parse_args()

    # 1. 加载 HeAR 模型
    print("📦 Loading HeAR Model...")
    model = tf.saved_model.load(args.hear_path)
    extract_fn = model.signatures["serving_default"]

    # 2. 分别处理 train 和 test 文件夹
    all_indices = {}
    for subset in ["train", "test"]:
        src_dir = Path(args.src_root) / subset
        dst_dir = Path(args.save_root) / subset

        if src_dir.exists():
            indices = process_subset(src_dir, dst_dir, extract_fn, subset)
            # 保存该子集的索引 CSV
            pd.DataFrame(indices).to_csv(Path(args.save_root) / f"{subset}_index.csv", index=False)
            print(f"✅ {subset} set processed. Index saved.")
        else:
            print(f"⚠️ Warning: {src_dir} not found, skipping.")

    print(f"\n🚀 All done! Tokens saved to {args.save_root}")


if __name__ == "__main__":
    main()