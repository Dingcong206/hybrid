#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import subprocess
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModel

# =========================
# 0) 默认路径与配置
# =========================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_multi_segments_npy")
OUT_CSV = os.path.join(BASE_DIR, "coswara_hear_multi_segments.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000
TARGET_LEN = 32000  # 2秒
HOP_SEC = 1.0  # 1秒步长
MIN_RMS = 0.005  # 能量阈值

LABEL_MAP = {
    "healthy": 0,
    "positive_mild": 1,
    "positive_moderate": 1,
    "positive_asymp": 1,
}

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1) 导入 HeAR 预处理函数
# =========================
def import_hear_preprocess():
    sys.path.insert(0, "/data/dingcong/hybrid")
    import importlib
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    return audio_utils.preprocess_audio


preprocess_audio = import_hear_preprocess()


# =========================
# 2) 基础工具函数
# =========================
def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), "-vn", tmp.name]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {p.stderr.strip()[:300]}")
        audio, _ = sf.read(tmp.name, dtype="float32")
        audio = torch.from_numpy(audio).float()
        return audio.unsqueeze(0) if audio.ndim == 1 else audio


def is_bad_name(fname: str) -> bool:
    return fname.startswith("._") or fname == ".DS_Store"


def get_all_segments(waveform: torch.Tensor, target_len: int, hop_size: int) -> list:
    wav = waveform.squeeze(0)
    T = wav.numel()
    if T <= target_len:
        return [F.pad(wav.unsqueeze(0), (0, target_len - T))]
    segments = []
    for i in range(0, T - target_len + 1, hop_size):
        seg = wav[i: i + target_len]
        if torch.sqrt((seg ** 2).mean()) > MIN_RMS:
            segments.append(seg.unsqueeze(0))
    if len(segments) == 0:
        mid = (T - target_len) // 2
        segments.append(wav[mid: mid + target_len].unsqueeze(0))
    return segments


# =========================
# 4) 主提取循环 (修正后的单循环逻辑)
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="调试用")
    args = parser.parse_args()

    print("🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    df_label = pd.read_csv(COSWARA_CSV)
    df_label["id"] = df_label["id"].astype(str).str.strip()
    label_lookup = dict(zip(df_label["id"], df_label["covid_status"]))

    audio_tasks = []
    for root, _, files in os.walk(BASE_DIR):
        if os.path.abspath(SAVE_DIR) in os.path.abspath(root): continue
        for f in files:
            if is_bad_name(f): continue
            if os.path.splitext(f)[1].lower() not in [".wav", ".webm"]: continue
            parts = root.split(os.sep)
            u_id = next((p for p in parts if len(p) > 20), None)
            if not u_id or label_lookup.get(u_id) not in LABEL_MAP: continue
            audio_tasks.append({
                "user_id": u_id, "path": os.path.join(root, f),
                "fname": f, "label": LABEL_MAP[label_lookup[u_id]], "status": label_lookup[u_id]
            })

    if args.limit > 0: audio_tasks = audio_tasks[:args.limit]
    print(f"📊 待处理文件: {len(audio_tasks)}")

    rows = []
    stats = {"success": 0, "total_segs": 0, "fail": 0, "skipped": 0}

    with torch.no_grad():
        for task in tqdm(audio_tasks):
            try:
                clean_fname = task['fname'].replace('.', '_')
                save_prefix = f"{task['user_id']}_{clean_fname}_seg"

                # --- 断点续传检查 ---
                first_seg = os.path.join(SAVE_DIR, f"{save_prefix}0.npy")
                if os.path.exists(first_seg):
                    existing = sorted(glob.glob(os.path.join(SAVE_DIR, f"{save_prefix}*.npy")))
                    for p in existing:
                        try:
                            idx = int(p.split('_seg')[-1].split('.npy')[0])
                        except:
                            idx = 0
                        rows.append({
                            "user_id": task["user_id"], "audio_file": task["fname"],
                            "segment_id": idx, "label": task["label"],
                            "feature_path": p, "covid_status": task["status"]
                        })
                        stats["total_segs"] += 1
                    stats["success"] += 1
                    stats["skipped"] += 1
                else:
                    # --- 正常提取逻辑 ---
                    wav = load_audio_ffmpeg(task["path"], TARGET_SR)
                    segments = get_all_segments(wav, TARGET_LEN, int(TARGET_SR * HOP_SEC))
                    for idx, wav2s in enumerate(segments):
                        spec = preprocess_audio(wav2s)
                        out = model(spec.to(DEVICE), return_dict=True, output_hidden_states=True)
                        feat = out.hidden_states[0][:, 1:, :].squeeze(0).cpu().numpy()

                        s_path = os.path.join(SAVE_DIR, f"{save_prefix}{idx}.npy")
                        np.save(s_path, feat)
                        rows.append({
                            "user_id": task["user_id"], "audio_file": task["fname"],
                            "segment_id": idx, "label": task["label"],
                            "feature_path": s_path, "covid_status": task["status"]
                        })
                        stats["total_segs"] += 1
                    stats["success"] += 1

                # 定期保存防止丢失
                if stats["success"] % 100 == 0:
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

            except Exception:
                stats["fail"] += 1

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\n✨ 完成! 成功: {stats['success']} (跳过: {stats['skipped']}), 总片段: {stats['total_segs']}")
    else:
        print("❌ 未提取到特征")


if __name__ == "__main__":
    main()