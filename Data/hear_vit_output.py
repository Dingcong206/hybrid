#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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

# -------------------------
# ICBHI label mapping (4-class)
# -------------------------
def lungsound_label(crackles: int, wheezes: int) -> int:
    c, w = int(crackles), int(wheezes)
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3

# -------------------------
# Audio utils
# -------------------------
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

# -------------------------
# Official split
# -------------------------
def load_official_split(split_file: Path) -> Dict[str, str]:
    """
    official_split.txt: <file_stem>\t<train|test>
    """
    df = pd.read_csv(split_file, sep="\t", header=None, names=["file", "set"])
    split_map = dict(zip(df["file"].astype(str), df["set"].astype(str)))
    return split_map

def get_set_for_recording(stem: str, split_map: Dict[str, str]) -> Optional[str]:
    if stem in split_map:
        return split_map[stem]
    if (stem + ".wav") in split_map:
        return split_map[stem + ".wav"]
    return None

# -------------------------
# Import HeAR preprocess_audio (from Google-Health/hear repo)
# -------------------------
def import_preprocess_audio(hear_repo_root: str):
    """
    HuggingFace model card uses:
      audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
      preprocess_audio = audio_utils.preprocess_audio
    我们做成“可控导入”：把 hear repo 根目录加到 sys.path
    """
    hear_repo_root = str(hear_repo_root)
    if hear_repo_root not in sys.path:
        sys.path.insert(0, hear_repo_root)

    try:
        import importlib
        audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
        preprocess_audio = audio_utils.preprocess_audio
        return preprocess_audio
    except Exception as e:
        raise RuntimeError(
            f"无法导入 hear.python.data_processing.audio_utils。\n"
            f"请确认你已经 git clone 了 Google-Health/hear，并且路径正确。\n"
            f"当前 hear_repo_root={hear_repo_root}\n"
            f"原始错误: {repr(e)}"
        )

# -------------------------
# HeAR embedding extraction (ViT后 embedding)
# -------------------------
@torch.no_grad()
def hear_embedding_from_waveform(model, preprocess_audio, wav_2s: torch.Tensor) -> Tuple[np.ndarray, str]:
    """
    修改后：提取完整的序列 Embedding (L, D)
    wav_2s: (1, 32000)
    返回: (L, 1024) 或 (L, 768) 的 numpy 数组
    """
    wav_cpu = wav_2s.detach().float().cpu()
    spec = preprocess_audio(wav_cpu)

    if isinstance(spec, dict):
        spec_in = spec["pixel_values"] if "pixel_values" in spec else next(iter(spec.values()))
    else:
        spec_in = spec

    spec_in = spec_in.to(next(model.parameters()).device)

    # 1. 运行模型前向传播
    out = model.forward(spec_in, return_dict=True, output_hidden_states=True)

    # 2. 提取序列特征 (Last Hidden State)
    # HeAR (ViT-based) 的 last_hidden_state 形状通常是 (Batch, Seq_Len, Hidden_Dim)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        emb = out.last_hidden_state  # 形状为 (1, L, D)
        src = "last_hidden_state_full_sequence"
    else:
        # 兜底逻辑：如果找不到 last_hidden_state，尝试从 hidden_states 获取最后一层
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            emb = out.hidden_states[-1]
            src = "hidden_states_last_layer"
        else:
            raise RuntimeError("无法找到序列特征，请检查模型输出字段")

    # 3. 去掉 Batch 维度，保留 (L, D)
    # 注意：ViT 通常包含一个 CLS token (index 0)，
    # 如果你不需要 CLS token，可以使用 emb[0, 1:, :]
    emb = emb.squeeze(0)

    return emb.detach().cpu().numpy().astype(np.float32), src

# -------------------------
# One recording -> cycles
# -------------------------
def process_recording(
    wav_path: Path,
    txt_path: Path,
    out_root: Path,
    split_name: str,
    model,
    preprocess_audio,
    device
) -> List[Dict]:
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

        emb, emb_src = hear_embedding_from_waveform(model, preprocess_audio, cycle_2s)

        save_subdir = out_root / split_name / patient_id
        save_subdir.mkdir(parents=True, exist_ok=True)

        npy_name = f"{recording}_cycle{i:04d}_y{y4}.npy"
        npy_path = save_subdir / npy_name
        np.save(npy_path, emb)

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

    # ICBHI wav/txt 所在目录
    parser.add_argument("--data_root", type=str, default="/data/dingcong/hybrid/audio_and_txt_files",
                        help="ICBHI wav/txt 根目录（*.wav + *.txt）")

    # 官方 split
    parser.add_argument("--split_file", type=str, default="/data/dingcong/hybrid/audio_and_txt_files/official_split.txt",
                        help="官方 train/test 划分文件")

    # 你 clone 的 Google-Health/hear repo 根目录（包含 hear/python/...）
    parser.add_argument("--hear_repo_root", type=str, default="/data/dingcong/hybrid/hear",
                        help="Google-Health/hear 仓库根目录（用于导入 preprocess_audio）")

    # 输出目录
    parser.add_argument("--out_dir", type=str, default="/data/dingcong/hybrid/icbhi_hear_vit_embedding_512",
                        help="输出目录（train/test 子目录 + index.csv）")

    # 本地 HF snapshot
    parser.add_argument("--model_dir", type=str,
                        default="/home/guest1/.cache/huggingface/hub/models--google--hear-pytorch/snapshots/f791cd42437c3e268c8ac84707e3508900f65f1a",
                        help="本地 google/hear-pytorch snapshot 路径")

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

    # split
    split_map = load_official_split(split_file)

    # device + model
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("loading model from:", args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    print("✅ model loaded")

    # preprocess_audio
    preprocess_audio = import_preprocess_audio(args.hear_repo_root)
    print("✅ preprocess_audio loaded from hear repo")

    # iterate recordings
    wav_files = sorted(data_root.glob("*.wav"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav found under {data_root}")

    train_rows: List[Dict] = []
    test_rows: List[Dict] = []
    skipped_no_split = 0
    skipped_no_txt = 0
    skipped_empty_cycles = 0

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
            preprocess_audio=preprocess_audio,
            device=device
        )
        if len(rows) == 0:
            skipped_empty_cycles += 1
            continue

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
    print(f"skipped: no_split={skipped_no_split}, no_txt={skipped_no_txt}, empty_cycles={skipped_empty_cycles}")

    # sanity
    if len(df_train) > 0:
        p = df_train.iloc[0]["tokens_path"]
        arr = np.load(p)
        print("example embedding shape:", arr.shape)
        print("example embedding_source:", df_train.iloc[0]["embedding_source"])
        print("example path:", p)

if __name__ == "__main__":
    main()
