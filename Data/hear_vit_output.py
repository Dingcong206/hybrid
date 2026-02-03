#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from tqdm import tqdm
from transformers import AutoModel

# =========================
# Constants
# =========================
SR = 16000
TARGET_SAMPLES = 32000  # 2 sec * 16k

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
# Audio utilities
# =========================
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
    - if long: choose 2s window with max energy (step 100ms)
    - if short: center pad
    """
    n = wav.shape[-1]
    if n > TARGET_SAMPLES:
        step = 1600  # 0.1s
        windows = wav.unfold(-1, TARGET_SAMPLES, step)  # (1, num_win, 32000)
        energies = torch.sum(windows ** 2, dim=-1)      # (1, num_win)
        best_idx = torch.argmax(energies, dim=-1).item()
        return windows[0, best_idx].unsqueeze(0)        # (1, 32000)
    elif n < TARGET_SAMPLES:
        pad_total = TARGET_SAMPLES - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(wav, (pad_left, pad_right))
    else:
        return wav

# =========================
# Official split loader
# =========================
def load_official_split(split_file: Path) -> Dict[str, str]:
    """
    official_split.txt 格式（你之前用过）：
      <file_stem>\t<train|test>
    可能是：
      101_1b1_Al_sc_Meditron\ttrain
    """
    df = pd.read_csv(split_file, sep="\t", header=None, names=["file", "set"])
    split_map = dict(zip(df["file"].astype(str), df["set"].astype(str)))
    return split_map

def get_set_for_recording(stem: str, split_map: Dict[str, str]) -> Optional[str]:
    """
    同时兼容 split_map 存的是 stem 或 filename 的情况
    """
    if stem in split_map:
        return split_map[stem]
    if (stem + ".wav") in split_map:
        return split_map[stem + ".wav"]
    return None

# =========================
# HeAR PyTorch embedding (ViT后)
# =========================
@torch.no_grad()
def hear_vit_embedding(model, wav_2s: torch.Tensor) -> Tuple[np.ndarray, str]:
    """
    wav_2s: torch Tensor (1, 32000) on device
    return embedding: (1024,)
    """
    out = model(wav_2s, return_dict=True)

    # 优先：pooler_output（如果模型定义了pooler）
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output.squeeze(0)  # (1024,)
        src = "pooler_output"
        return emb.detach().cpu().numpy(), src

    # 兜底：CLS token = last_hidden_state[:,0,:]
    if not hasattr(out, "last_hidden_state") or out.last_hidden_state is None:
        raise RuntimeError("No pooler_output and no last_hidden_state. Cannot get embedding.")
    emb = out.last_hidden_state[:, 0, :].squeeze(0)  # (1024,)
    src = "cls_from_last_hidden_state"
    return emb.detach().cpu().numpy(), src

# =========================
# Per-recording cycle extraction
# =========================
def process_recording(
    wav_path: Path,
    txt_path: Path,
    out_root: Path,
    split_name: str,
    model,
    device
) -> List[Dict]:
    """
    对一个 recording：
      - 读取 wav
      - 读取 txt cycles
      - 对每个 cycle: crop/pad 2s -> vit embedding -> save npy
    返回 rows（用于 index.csv）
    """
    wav = load_wav_resample_mono(str(wav_path))  # (1,N)
    ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])

    recording = wav_path.stem
    patient_id = recording.split("_")[0]

    rows: List[Dict] = []

    for i, r in ann.iterrows():
        start_i = int(float(r["Start"]) * SR)
        end_i   = int(float(r["End"])   * SR)
        cycle = wav[:, start_i:end_i]
        if cycle.shape[-1] < 1600:
            continue

        y4 = lungsound_label(r["C"], r["W"])
        cycle_2s = process_to_2s_sample(cycle).to(device).float()  # (1,32000)

        emb, emb_src = hear_vit_embedding(model, cycle_2s)          # (1024,)

        save_subdir = out_root / split_name / patient_id
        save_subdir.mkdir(parents=True, exist_ok=True)

        npy_name = f"{recording}_cycle{i:04d}_y{y4}.npy"
        npy_path = save_subdir / npy_name
        np.save(npy_path, emb.astype(np.float32))

        rows.append({
            "tokens_path": str(npy_path),
            "label": int(y4),
            "recording": recording,
            "patient_id": patient_id,
            "set": split_name,
            "cycle_idx": int(i),
            "start_sec": float(r["Start"]),
            "end_sec": float(r["End"]),
            "embedding_source": emb_src,
            "shape": str(emb.shape),
            "wav_path": str(wav_path),
            "txt_path": str(txt_path),
        })

    return rows

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # 你的 ICBHI wav/txt 在这里（不要再指向 ast tokens 目录）
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/dingcong/hybrid/audio_and_txt_files",
        help="ICBHI wav/txt 根目录（里面直接放 *.wav + *.txt）"
    )

    # 官方划分文件（你之前就用过）
    parser.add_argument(
        "--split_file",
        type=str,
        default="/data/dingcong/hybrid/audio_and_txt_files/official_split.txt",
        help="官方 train/test 划分文件"
    )

    # 输出：vit embedding
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/dingcong/hybrid/icbhi_hear_vit_embedding_1024",
        help="输出目录（会生成 train/test 子目录 + index.csv）"
    )

    # 你本地 cache 的 hear-pytorch snapshot（你已经有了）
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/guest1/.cache/huggingface/hub/models--google--hear-pytorch/snapshots/f791cd42437c3e268c8ac84707e3508900f65f1a",
        help="本地 google/hear-pytorch snapshot 路径"
    )

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    out_root = Path(args.out_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_file not found: {split_file}")

    out_root.mkdir(parents=True, exist_ok=True)

    # load split
    split_map = load_official_split(split_file)

    # device + model
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("loading model from:", args.model_dir)

    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    print("✅ model loaded")

    # list all recordings from data_root/*.wav
    wav_files = sorted(data_root.glob("*.wav"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav found under {data_root}. Expected wav/txt in this folder.")

    train_rows: List[Dict] = []
    test_rows: List[Dict] = []
    skipped_no_split = 0
    skipped_no_txt = 0

    for wav_path in tqdm(wav_files, desc="Recordings"):
        recording = wav_path.stem
        split_name = get_set_for_recording(recording, split_map)
        if split_name not in ("train", "test"):
            skipped_no_split += 1
            continue

        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            skipped_no_txt += 1
            continue

        rows = process_recording(
            wav_path=wav_path,
            txt_path=txt_path,
            out_root=out_root,
            split_name=split_name,
            model=model,
            device=device
        )

        if split_name == "train":
            train_rows.extend(rows)
        else:
            test_rows.extend(rows)

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    train_index = out_root / "train_index.csv"
    test_index = out_root / "test_index.csv"
    df_train.to_csv(train_index, index=False)
    df_test.to_csv(test_index, index=False)

    print("\n✨ Done.")
    print(f"train cycles: {len(df_train)} | test cycles: {len(df_test)}")
    print("saved:", train_index, test_index)
    print(f"skipped: no_split={skipped_no_split}, no_txt={skipped_no_txt}")

    # quick sanity print
    if len(df_train) > 0:
        p = df_train.iloc[0]["tokens_path"]
        arr = np.load(p)
        print("example embedding shape:", arr.shape, "path:", p)

if __name__ == "__main__":
    main()
