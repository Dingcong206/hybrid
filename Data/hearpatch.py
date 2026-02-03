#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms as T
import tensorflow as tf  # HeAR 使用 TensorFlow
from tqdm import tqdm

# =========================
# HeAR-aligned constants
# =========================
SR = 16000
DURATION_SEC = 2  # HeAR 标准是 2s
TARGET_SAMPLES = SR * DURATION_SEC  # 32000 samples


# =========================
# 4-class label mapping
# =========================
def lungsound_label(crackles: int, wheezes: int) -> int:
    crackles, wheezes = int(crackles), int(wheezes)
    if crackles == 0 and wheezes == 0: return 0
    if crackles == 1 and wheezes == 0: return 1
    if crackles == 0 and wheezes == 1: return 2
    return 3


# =========================
# Audio Utilities (与你之前的 AST 脚本一致)
# =========================
def load_wav_resample_mono(wav_path: str, target_sr: int = SR) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = T.Resample(orig_sr, target_sr)(wav)
    return wav


def fix_to_2s_trunc_or_repeat(x: torch.Tensor) -> torch.Tensor:
    """HeAR 专用：调整到 2s (32000个采样点)"""
    if x.shape[-1] >= TARGET_SAMPLES:
        y = x[..., :TARGET_SAMPLES]
    else:
        # 循环填充
        import math
        ratio = math.ceil(TARGET_SAMPLES / max(1, x.shape[-1]))
        y = x.repeat(1, ratio)[..., :TARGET_SAMPLES]
    return y


# =========================
# HeAR 特征提取 (替换 AST 的核心)
# =========================
def extract_hear_patch_tokens(model_fn, wav_2s: torch.Tensor) -> np.ndarray:
    """
    输入: (1, 32000) Torch Tensor
    输出: (128, 1024) Numpy Array (进入 ViT 之前的 Patch Tokens)
    """
    # 转为 TF 张量格式
    audio_tf = tf.constant(wav_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)
    # 调用 HeAR 签名接口
    outputs = model_fn(audio_wav=audio_tf)
    # 提取 Patch Embeddings (通常是输出字典中的第一个值)
    # 形状应为 [1, 128, 1024] -> squeeze 得到 [128, 1024]
    tokens = list(outputs.values())[0].numpy().squeeze(0)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    # 路径完全沿用你 AST 脚本的设置
    parser.add_argument("--data_dir", type=str, default="/data/dingcong/hybrid/audio_and_txt_files")
    parser.add_argument("--out_dir", type=str, default="/data/dingcong/hybrid/icbhi_hear_patch_128_1024")
    parser.add_argument("--hear_path", type=str,
                        default="/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f/event_detector/spectrogram_frontend")
    args = parser.parse_args()

    # 1. 加载 HeAR 模型
    print(f"📦 Loading HeAR Model from: {args.hear_path}")
    hear_model = tf.saved_model.load(args.hear_path)
    extract_fn = hear_model.signatures["serving_default"]

    # 2. 自动定位官方划分文件 (沿用你 AST 脚本逻辑)
    # 这里为了简便，假设你已经知道 split 文件位置或使用你 AST 的 find_official_split_file
    from pathlib import Path
    split_path = os.path.join(args.data_dir, "official_split.txt")
    # 如果不存在，请手动修正此路径
    split_df = pd.read_csv(split_path, sep='\t', names=['file', 'set'])
    split_map = dict(zip(split_df['file'], split_df['set']))

    # 3. 创建目录
    for s in ["train", "test"]: os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)

    # 4. 扫描文件并处理
    wav_files = list(Path(args.data_dir).glob("*.wav"))
    all_rows = []

    for wav_path in tqdm(wav_files):
        tag = split_map.get(wav_path.stem) or split_map.get(wav_path.name)
        if not tag: continue

        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists(): continue

        wav = load_wav_resample_mono(str(wav_path))
        ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])
        patient_id = wav_path.stem.split("_")[0]

        for i, r in ann.iterrows():
            start_i, end_i = int(r["Start"] * SR), int(r["End"] * SR)
            cycle = wav[:, start_i:end_i]
            if cycle.shape[-1] < 1600: continue

            # 统一到 2s
            cycle_2s = fix_to_2s_trunc_or_repeat(cycle)

            # 提取 HeAR Tokens (128, 1024)
            tokens = extract_hear_patch_tokens(extract_fn, cycle_2s)

            # 保存
            save_subdir = os.path.join(args.out_dir, tag, patient_id)
            os.makedirs(save_subdir, exist_ok=True)
            npy_name = f"{wav_path.stem}_c{i}_y{lungsound_label(r['C'], r['W'])}.npy"
            npy_path = os.path.join(save_subdir, npy_name)
            np.save(npy_path, tokens)

            all_rows.append({
                "tokens_path": npy_path,
                "label": lungsound_label(r["C"], r["W"]),
                "patient_id": patient_id,
                "set": tag
            })

    # 5. 保存索引 CSV
    full_df = pd.DataFrame(all_rows)
    for s in ["train", "test"]:
        full_df[full_df['set'] == s].to_csv(os.path.join(args.out_dir, f"{s}_index.csv"), index=False)

    print(f"🚀 DONE! Tokens saved to {args.out_dir}. Shape: {tokens.shape}")


if __name__ == "__main__":
    main()