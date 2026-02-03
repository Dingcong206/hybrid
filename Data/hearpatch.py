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
import tensorflow as tf
from tqdm import tqdm

# =========================
# HeAR-aligned constants
# =========================
SR = 16000
DURATION_SEC = 2
TARGET_SAMPLES = SR * DURATION_SEC  # 32000 samples


def lungsound_label(crackles: int, wheezes: int) -> int:
    crackles, wheezes = int(crackles), int(wheezes)
    if crackles == 0 and wheezes == 0: return 0
    if crackles == 1 and wheezes == 0: return 1
    if crackles == 0 and wheezes == 1: return 2
    return 3


def load_wav_resample_mono(wav_path: str, target_sr: int = SR) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = T.Resample(orig_sr, target_sr)(wav)
    return wav


def fix_to_2s_trunc_or_repeat(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] >= TARGET_SAMPLES:
        y = x[..., :TARGET_SAMPLES]
    else:
        import math
        ratio = math.ceil(TARGET_SAMPLES / max(1, x.shape[-1]))
        y = x.repeat(1, ratio)[..., :TARGET_SAMPLES]
    return y


# =========================
# HeAR 特征提取 (进入 ViT 之前的 Patch Tokens)
# =========================
def extract_hear_patch_tokens(model_fn, wav_2s: torch.Tensor) -> np.ndarray:
    """
    输入: (1, 32000) Torch Tensor
    输出: (128, 1024) Numpy Array (Patch Projection 后, 进入 ViT 前)
    """
    # 转为 TF 张量
    audio_tf = tf.constant(wav_2s.numpy().reshape(1, TARGET_SAMPLES), dtype=tf.float32)

    # 调用 get_patch_embeddings 签名
    # 这会输出 {'patch_embeddings': <Tensor shape=(1, 128, 1024)>}
    outputs = model_fn(audio_wav=audio_tf)

    # 提取结果并去掉 batch 维度 -> (128, 1024)
    tokens = outputs['patch_embeddings'].numpy().squeeze(0)
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/dingcong/hybrid/audio_and_txt_files")
    parser.add_argument("--out_dir", type=str, default="/data/dingcong/hybrid/icbhi_hear_patch_128_1024")

    # 【关键修改】路径指向 encoder 目录
    parser.add_argument("--hear_path", type=str,
                        default="/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f/event_detector/encoder")
    args = parser.parse_args()

    # 1. 加载 HeAR Encoder 模块
    print(f"📦 Loading HeAR Encoder from: {args.hear_path}")
    hear_model = tf.saved_model.load(args.hear_path)

    # 【关键修改】使用 get_patch_embeddings 签名
    if "get_patch_embeddings" in hear_model.signatures:
        extract_fn = hear_model.signatures["get_patch_embeddings"]
        print("✅ 成功关联 'get_patch_embeddings' 签名 (输出维度: 1024)")
    else:
        # 兜底方案：打印所有签名供检查
        print(f"❌ 未找到指定签名。可用签名: {list(hear_model.signatures.keys())}")
        return

    # 2. 官方划分文件
    split_path = os.path.join(args.data_dir, "official_split.txt")
    split_df = pd.read_csv(split_path, sep='\t', names=['file', 'set'])
    split_map = dict(zip(split_df['file'], split_df['set']))

    # 3. 创建目录
    for s in ["train", "test"]: os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)

    # 4. 扫描并处理
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

            cycle_2s = fix_to_2s_trunc_or_repeat(cycle)

            # 提取维度为 (128, 1024) 的 Patch Tokens
            tokens = extract_hear_patch_tokens(extract_fn, cycle_2s)

            # 按患者 ID 分文件夹保存
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

    # 5. 保存索引
    full_df = pd.DataFrame(all_rows)
    for s in ["train", "test"]:
        full_df[full_df['set'] == s].to_csv(os.path.join(args.out_dir, f"{s}_index.csv"), index=False)

    print(f"🚀 处理完成！最终特征维度: {tokens.shape}")


if __name__ == "__main__":
    main()