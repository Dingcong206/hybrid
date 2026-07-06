#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms as T

from transformers import ASTModel


# ============================================================
# ✅ 写死路径（你要改路径只改这里两行）
# ============================================================
DATA_DIR = "/data/dingcong/hybrid/audio_and_txt_files"          # 你的 ICBHI wav+txt 目录
OUT_DIR  = "/data/dingcong/hybrid/icbhi_official_fbank"         # 输出目录（train/test + csv）

# =========================
# Paper-aligned constants
# =========================
SR = 16000
DURATION_SEC = 8
TARGET_SAMPLES = SR * DURATION_SEC          # 128000
N_MELS = 128
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
TARGET_FRAMES = 798

FBANK_MEAN = -4.27
FBANK_STD = 4.57


# =========================
# 4-class label mapping
# 0=normal,1=crackle,2=wheeze,3=both
# =========================
def lungsound_label(crackles: int, wheezes: int) -> int:
    crackles = int(crackles)
    wheezes = int(wheezes)
    if crackles == 0 and wheezes == 0:
        return 0
    if crackles == 1 and wheezes == 0:
        return 1
    if crackles == 0 and wheezes == 1:
        return 2
    return 3


# =========================
# Split file utilities
# =========================
def find_official_split_file(data_dir: str) -> str:
    """
    Try to locate official_split.txt in common places.
    """
    p = Path(data_dir).resolve()
    candidates = [
        p / "official_split.txt",
        p / "icbhi_dataset" / "official_split.txt",
        p.parent / "official_split.txt",
        p.parent / "icbhi_dataset" / "official_split.txt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    for c in list(p.rglob("official_split.txt"))[:20]:
        return str(c)

    raise FileNotFoundError(
        "找不到 official_split.txt。\n"
        "请把 official_split.txt 放到以下任意位置之一：\n"
        f"1) {p}/official_split.txt\n"
        f"2) {p}/icbhi_dataset/official_split.txt\n"
        f"3) {p.parent}/official_split.txt\n"
        f"4) {p.parent}/icbhi_dataset/official_split.txt\n"
    )


def load_official_split(split_path: str) -> Dict[str, str]:
    """
    Robust matching:
    - Accept keys as: 'xxx.wav' OR 'xxx' OR 'path/to/xxx.wav'
    - Store BOTH: basename-with-ext AND stem-without-ext
    """
    mapping: Dict[str, str] = {}
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue

            raw = parts[0].strip()
            tag = parts[1].strip().lower()
            if tag not in ("train", "test"):
                continue

            name = Path(raw).name
            stem = Path(name).stem

            mapping[name] = tag
            mapping[stem] = tag
            if "." not in name:
                mapping[f"{name}.wav"] = tag

    if not mapping:
        raise ValueError(f"official_split.txt 解析为空：{split_path}")
    return mapping


# =========================
# Audio + cycle parsing
# =========================
def load_wav_resample_mono(wav_path: str, target_sr: int = SR) -> torch.Tensor:
    """
    Returns: (1, T) float32, mono, resampled to target_sr.
    """
    wav, orig_sr = torchaudio.load(wav_path)  # (C, T)
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        wav = T.Resample(orig_sr, target_sr)(wav)
    return wav


def apply_fade_in_out(x: torch.Tensor, fade_samples: int) -> torch.Tensor:
    fade_samples = max(1, min(fade_samples, x.shape[-1]))
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear")
    return fade(x)


def slice_cycle(wav: torch.Tensor, start_s: float, end_s: float, sr: int = SR) -> torch.Tensor:
    """
    wav: (1, T)
    """
    T_total = wav.shape[-1]
    start_i = int(max(0.0, float(start_s)) * sr)
    end_i = int(max(0.0, float(end_s)) * sr)
    start_i = min(start_i, T_total)
    end_i = min(end_i, T_total)
    if end_i <= start_i:
        return torch.zeros(1, 0, dtype=wav.dtype)
    return wav[..., start_i:end_i]


def fix_to_8s_trunc_or_repeat_fade(x: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """
    Paper-aligned:
    - longer: truncate
    - shorter: repeat until >= target, then truncate
    - fade in/out on final clip
    """
    assert x.dim() == 2 and x.shape[0] == 1, "x should be (1, T)"
    if x.shape[-1] >= TARGET_SAMPLES:
        y = x[..., :TARGET_SAMPLES]
    else:
        ratio = math.ceil(TARGET_SAMPLES / max(1, x.shape[-1]))
        y = x.repeat(1, ratio)[..., :TARGET_SAMPLES]

    fade_samples = int(sr / 16)  # ~62.5ms at 16kHz
    y = apply_fade_in_out(y, fade_samples)
    return y


def read_cycle_txt(txt_path: str) -> pd.DataFrame:
    """
    ICBHI txt format (tab-delimited):
    Start  End  Crackles  Wheezes
    """
    df = pd.read_csv(txt_path, sep="\t", header=None, names=["Start", "End", "Crackles", "Wheezes"])
    df["Start"] = df["Start"].astype(float)
    df["End"] = df["End"].astype(float)
    df["Crackles"] = df["Crackles"].astype(int)
    df["Wheezes"] = df["Wheezes"].astype(int)
    return df


# =========================
# Fbank extraction (paper-aligned)
# =========================
def wav_to_fbank_798x128(wav_8s: torch.Tensor) -> np.ndarray:
    """
    wav_8s: (1, 128000)
    return: (798, 128) float32, paper normalization
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        wav_8s,
        htk_compat=True,
        sample_frequency=SR,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=N_MELS,
        dither=0.0,
        frame_length=FRAME_LENGTH_MS,  # 25ms
        frame_shift=FRAME_SHIFT_MS     # 10ms
    )  # (T, 128)

    # paper normalization
    fbank = (fbank - FBANK_MEAN) / FBANK_STD

    # force frames to 798
    T_frames = fbank.shape[0]
    if T_frames > TARGET_FRAMES:
        fbank = fbank[:TARGET_FRAMES]
    elif T_frames < TARGET_FRAMES:
        pad = torch.zeros((TARGET_FRAMES - T_frames, fbank.shape[1]),
                          dtype=fbank.dtype, device=fbank.device)
        fbank = torch.cat([fbank, pad], dim=0)

    return fbank.cpu().numpy().astype(np.float32)  # (798, 128)


# =========================
# Pair finding
# =========================
def find_wav_txt_pairs(data_dir: str) -> List[Tuple[str, str]]:
    root = Path(data_dir)
    wavs = {p.stem: str(p) for p in root.glob("*.wav")}
    txts = {p.stem: str(p) for p in root.glob("*.txt")}
    stems = sorted(set(wavs.keys()) & set(txts.keys()))
    return [(wavs[s], txts[s]) for s in stems]


# =========================
# Main pipeline (save FBANK; tokens optional)
# =========================
def process_recording_to_features(
    wav_path: str,
    txt_path: str,
    out_dir: str,
    save_fbank: bool = True,
    save_tokens: bool = False,
    ast_model: Optional[ASTModel] = None,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """
    For one recording:
    wav+txt -> cycle slicing -> 8s fix -> fbank -> save fbank.npy (primary)
    optional: also save tokens.npy for DEBUG only (fixed features)
    """
    wav = load_wav_resample_mono(wav_path, SR)
    wav = apply_fade_in_out(wav, int(SR / 16))  # slight fade on whole recording

    ann = read_cycle_txt(txt_path)
    base = Path(wav_path).stem
    rows: List[dict] = []

    for i, r in ann.iterrows():
        cycle = slice_cycle(wav, r["Start"], r["End"], SR)
        if cycle.shape[-1] < 10:
            continue

        y = lungsound_label(r["Crackles"], r["Wheezes"])
        cycle_8s = fix_to_8s_trunc_or_repeat_fade(cycle, SR)  # (1, 128000)

        fbank = wav_to_fbank_798x128(cycle_8s)                # (798, 128) np.float32

        fb_path = ""
        if save_fbank:
            fb_name = f"{base}_cycle{i:04d}_y{y}_fbank.npy"
            fb_path = os.path.join(out_dir, fb_name)
            np.save(fb_path, fbank)

        tok_path = ""
        tok_shape = ""
        if save_tokens:
            if ast_model is None or device is None:
                raise ValueError("save_tokens=True 需要提供 ast_model 和 device")
            # tokens only for debugging (no grad, fixed)
            with torch.no_grad():
                # (798,128) -> (1,798,128)
                fb_t = torch.from_numpy(fbank).unsqueeze(0).to(device)  # (1, 798, 128)
                # Conv2d projection expects (B,1,128,798)
                x = fb_t.transpose(1, 2).unsqueeze(1)                  # (1,1,128,798)
                conv = ast_model.embeddings.patch_embeddings.projection
                y_tok = conv(x).flatten(2).transpose(1, 2)[0]          # (N, hidden)
                tok = y_tok.cpu().numpy().astype(np.float32)
            tok_name = f"{base}_cycle{i:04d}_y{y}_tokens.npy"
            tok_path = os.path.join(out_dir, tok_name)
            np.save(tok_path, tok)
            tok_shape = str(tok.shape)

        rows.append({
            "recording": base,
            "cycle_index": int(i),
            "start_s": float(r["Start"]),
            "end_s": float(r["End"]),
            "crackles": int(r["Crackles"]),
            "wheezes": int(r["Wheezes"]),
            "label": int(y),
            "fbank_path": fb_path,
            "fbank_shape": str(fbank.shape),
            "tokens_path": tok_path,
            "tokens_shape": tok_shape,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    # ✅ data_dir/out_dir 已写死，不再作为参数
    parser.add_argument("--split_file", type=str, default="",
                        help="可选：手动指定 official_split.txt 路径；不填则自动搜索")
    parser.add_argument("--save_tokens", action="store_true", default=False,
                        help="（可选）额外保存 tokens.npy，仅用于调试/对照（不用于微调 projection）")
    parser.add_argument("--ast_model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593",
                        help="HuggingFace AST 模型名或本地路径（save_tokens 时才需要）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu（save_tokens 时才需要）")
    parser.add_argument("--local_files_only", action="store_true",
                        help="只从本地缓存加载 HF 模型（服务器不能联网时用）")
    args = parser.parse_args()

    data_dir = DATA_DIR
    out_dir = OUT_DIR

    # output
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "train")
    test_out = os.path.join(out_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    # official split
    split_path = args.split_file.strip() if args.split_file.strip() else find_official_split_file(data_dir)
    split_map = load_official_split(split_path)
    print(f"[INFO] official split file: {split_path} (keys={len(split_map)})")

    # pairs
    pairs = find_wav_txt_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"在 {data_dir} 没找到 wav/txt 配对文件。")
    print(f"[INFO] found wav/txt pairs: {len(pairs)}")

    # debug: check intersection quickly
    wav_names = [Path(w).name for w, _ in pairs]
    hit = sum([(n in split_map) or (Path(n).stem in split_map) for n in wav_names])
    print(f"[DEBUG] split match hits: {hit}/{len(wav_names)} (should be close to total)")

    # load AST only if saving tokens (debug)
    ast = None
    device = None
    if args.save_tokens:
        if args.device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            print("[WARN] cuda 不可用，自动切换到 cpu")
        else:
            device = torch.device(args.device)

        print(f"[INFO] loading AST model (for debug tokens): {args.ast_model}")
        ast = ASTModel.from_pretrained(args.ast_model, local_files_only=args.local_files_only)
        ast.eval().to(device)

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    skipped_no_split = 0

    for wav_path, txt_path in pairs:
        wav_name = Path(wav_path).name
        wav_stem = Path(wav_path).stem

        tag = split_map.get(wav_name, None)
        if tag is None:
            tag = split_map.get(wav_stem, None)

        if tag is None:
            skipped_no_split += 1
            continue

        out_subdir = train_out if tag == "train" else test_out

        rows = process_recording_to_features(
            wav_path=wav_path,
            txt_path=txt_path,
            out_dir=out_subdir,
            save_fbank=True,              # ✅ 始终保存 fbank（训练时在线投影）
            save_tokens=args.save_tokens, # 可选调试
            ast_model=ast,
            device=device,
        )

        if tag == "train":
            train_rows.extend(rows)
        else:
            test_rows.extend(rows)

    # save indexes
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_csv = os.path.join(out_dir, "train_index.csv")
    test_csv = os.path.join(out_dir, "test_index.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[DONE] Train cycles: {len(train_df)} -> {train_out}")
    print(f"[DONE] Test  cycles: {len(test_df)}  -> {test_out}")
    print(f"[DONE] train_index.csv: {train_csv}")
    print(f"[DONE] test_index.csv : {test_csv}")

    if skipped_no_split > 0:
        print(f"[WARN] skipped {skipped_no_split} recordings (not listed in official_split.txt after normalization)")

    if len(train_df) > 0:
        print("\n[STATS] Train label counts:")
        print(train_df["label"].value_counts().sort_index())
        print("[STATS] Example fbank shape:", train_df.iloc[0]["fbank_shape"])
        print("[STATS] Example fbank path :", train_df.iloc[0]["fbank_path"])
    if len(test_df) > 0:
        print("\n[STATS] Test label counts:")
        print(test_df["label"].value_counts().sort_index())
        print("[STATS] Example fbank shape:", test_df.iloc[0]["fbank_shape"])
        print("[STATS] Example fbank path :", test_df.iloc[0]["fbank_path"])


if __name__ == "__main__":
    main()