#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

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
TARGET_SAMPLES = 32000  # 2s * 16k

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
    n = wav.shape[-1]
    if n > TARGET_SAMPLES:
        step = 1600  # 0.1s
        windows = wav.unfold(-1, TARGET_SAMPLES, step)  # (1, num_win, 32000)
        energies = torch.sum(windows ** 2, dim=-1)      # (1, num_win)
        best_idx = torch.argmax(energies, dim=-1).item()
        return windows[0, best_idx].unsqueeze(0)
    elif n < TARGET_SAMPLES:
        pad_total = TARGET_SAMPLES - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(wav, (pad_left, pad_right))
    else:
        return wav

# =========================
# HeAR PyTorch: ViT embedding
# =========================
@torch.no_grad()
def hear_vit_embedding(model, wav_2s: torch.Tensor) -> Tuple[np.ndarray, str]:
    out = model(wav_2s, return_dict=True)

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output.squeeze(0)
        source = "pooler_output"
    else:
        if not hasattr(out, "last_hidden_state") or out.last_hidden_state is None:
            raise RuntimeError("No pooler_output and no last_hidden_state. Cannot get embedding.")
        emb = out.last_hidden_state[:, 0, :].squeeze(0)
        source = "cls_from_last_hidden_state"

    return emb.detach().cpu().numpy(), source

# =========================
# Main pipeline
# =========================
def process_split(split_name: str, split_dir: Path, out_dir: Path, model, device) -> pd.DataFrame:
    wav_files = sorted(split_dir.glob("*.wav"))
    rows: List[Dict] = []

    for wav_path in tqdm(wav_files, desc=f"Extract {split_name}"):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        patient_id = wav_path.stem.split("_")[0]
        wav = load_wav_resample_mono(str(wav_path))
        ann = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "C", "W"])

        for i, r in ann.iterrows():
            start_i = int(float(r["Start"]) * SR)
            end_i   = int(float(r["End"]) * SR)
            cycle = wav[:, start_i:end_i]

            if cycle.shape[-1] < 1600:
                continue

            y4 = lungsound_label(r["C"], r["W"])
            cycle_2s = process_to_2s_sample(cycle).to(device).float()

            emb, emb_src = hear_vit_embedding(model, cycle_2s)

            save_subdir = out_dir / split_name / patient_id
            save_subdir.mkdir(parents=True, exist_ok=True)

            npy_name = f"{wav_path.stem}_cycle{i}_y{y4}.npy"
            npy_path = save_subdir / npy_name
            np.save(npy_path, emb.astype(np.float32))

            rows.append({
                "tokens_path": str(npy_path),
                "label": int(y4),
                "recording": wav_path.stem,
                "patient_id": patient_id,
                "set": split_name,
                "embedding_source": emb_src,
                "shape": str(emb.shape),
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()

    # ✅【关键】我把你想要的路径写成默认值
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/dingcong/hybrid/audio_and_txt_files",
        help="包含 train/ 和 test/ 的数据目录"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/dingcong/hybrid/icbhi_hear_vit_embedding_1024",
        help="输出目录"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/guest1/.cache/huggingface/hub/models--google--hear-pytorch/snapshots/f791cd42437c3e268c8ac84707e3508900f65f1a",
        help="本地 HeAR PyTorch snapshot"
    )

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/test under {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("loading model from:", args.model_dir)

    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    print("✅ model loaded")

    df_train = process_split("train", train_dir, out_dir, model, device)
    df_test = process_split("test", test_dir, out_dir, model, device)

    train_index = out_dir / "train_index.csv"
    test_index = out_dir / "test_index.csv"
    df_train.to_csv(train_index, index=False)
    df_test.to_csv(test_index, index=False)

    print("\n✨ Done.")
    print("train n:", len(df_train), "test n:", len(df_test))
    print("saved:", train_index, test_index)

if __name__ == "__main__":
    main()
