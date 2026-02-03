#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import tensorflow as tf
from tqdm import tqdm

# ============================
# 配置与常量
# ============================
SR = 16000
TARGET_2S_SAMPLES = 32000
INTERNAL_PATCHES = 16
IN_DIM = 48
OUT_DIM = 256


# ============================
# HeAR 专用组件
# ============================
class PatchCompressor(nn.Module):
    def __init__(self, in_dim=48, out_dim=256, patches=16):
        super().__init__()
        self.patches = patches
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [B, 200, 48]
        x = x.permute(0, 2, 1)  # [B, 48, 200]
        x = F.interpolate(x, size=self.patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [B, 16, 48]
        return self.proj(x)  # [B, 16, 256]


def lungsound_label(crackles: int, wheezes: int) -> int:
    if crackles == 0 and wheezes == 0: return 0
    if crackles == 1 and wheezes == 0: return 1
    if crackles == 0 and wheezes == 1: return 2
    return 3


# ============================
# 音频处理工具
# ============================
def fix_to_2s_energy_based(x: torch.Tensor) -> torch.Tensor:
    """截取能量最高的核心 2s 或填充"""
    if x.shape[-1] > TARGET_2S_SAMPLES:
        # 滑动窗口查找最高能量片段 (步长 100ms)
        step = 1600
        windows = x.unfold(-1, TARGET_2S_SAMPLES, step)
        energies = torch.sum(windows ** 2, dim=-1)
        best_idx = torch.argmax(energies)
        return windows[0, best_idx].unsqueeze(0)
    else:
        pad_width = TARGET_2S_SAMPLES - x.shape[-1]
        return F.pad(x, (pad_width // 2, pad_width - pad_width // 2))


# ============================
# 主处理逻辑
# ============================
def process_recording(wav_path, txt_path, frontend_fn, compressor, device):
    # 加载并归一化音频
    wav, orig_sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != SR: wav = torchaudio.transforms.Resample(orig_sr, SR)(wav)

    # 解析标注
    ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])
    base = Path(wav_path).stem
    results = []

    for i, r in ann.iterrows():
        # 截取 Cycle
        start_i, end_i = int(r["Start"] * SR), int(r["End"] * SR)
        cycle = wav[:, start_i:end_i]
        if cycle.shape[-1] < 1600: continue  # 过滤短于 0.1s 的异常

        # 标准化为 2s (能量筛选)
        cycle_2s = fix_to_2s_energy_based(cycle)
        y = lungsound_label(r["C"], r["W"])

        # HeAR 提取
        audio_tf = tf.constant(cycle_2s.numpy().reshape(1, TARGET_2S_SAMPLES), dtype=tf.float32)
        out_tf = frontend_fn(audio_wav=audio_tf)
        spec_np = list(out_tf.values())[0][0].numpy()  # [200, 48]

        # Compressor 提取 Tokens
        spec_pt = torch.from_numpy(spec_np).unsqueeze(0).to(device)
        with torch.no_grad():
            tokens = compressor(spec_pt)  # [1, 16, 256]

        results.append({
            "tokens": tokens.squeeze(0).cpu().numpy(),
            "label": y,
            "cycle_idx": i
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/dingcong/hybrid/audio_and_txt_files")
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/hear_tokens")
    parser.add_argument("--hear_path", type=str, default="/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f/event_detector/spectrogram_frontend")
    parser.add_argument("--split_file", type=str, default="/data/dingcong/hybrid/audio_and_txt_files/official_split")
    args = parser.parse_args()

    # 1. 初始化模型
    print("📦 Loading Models...")
    frontend = tf.saved_model.load(args.hear_path)
    frontend_fn = frontend.signatures["serving_default"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compressor = PatchCompressor(IN_DIM, OUT_DIM, INTERNAL_PATCHES).to(device).eval()

    # 2. 加载划分
    split_df = pd.read_csv(args.split_file, sep='\t', names=['file', 'set'])
    split_map = dict(zip(split_df['file'], split_df['set']))

    # 3. 遍历文件
    wavs = list(Path(args.data_dir).glob("*.wav"))
    for wav_path in tqdm(wavs):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists(): continue

        tag = split_map.get(wav_path.stem) or split_map.get(wav_path.name)
        if not tag: continue

        # 处理并获取结果
        cycle_results = process_recording(wav_path, txt_path, frontend_fn, compressor, device)

        # 保存结果（按患者分子目录）
        patient_id = wav_path.stem.split("_")[0]
        out_path = Path(args.save_dir) / tag / patient_id
        out_path.mkdir(parents=True, exist_ok=True)

        for res in cycle_results:
            fname = f"{wav_path.stem}_c{res['cycle_idx']}_y{res['label']}.npy"
            np.save(out_path / fname, res['tokens'])

    print(f"✅ 处理完成！数据已保存在: {args.save_dir}")


if __name__ == "__main__":
    main()