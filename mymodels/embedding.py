#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import subprocess

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

# 更改输出路径以区分单窗版本
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_multi_segments_npy")
OUT_CSV = os.path.join(BASE_DIR, "coswara_hear_multi_segments.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000
TARGET_LEN = 32000  # 2秒 (16000 * 2)
HOP_SEC = 1.0  # 滑动步长：1秒 (即50%重叠)
MIN_RMS = 0.005  # 能量阈值：过滤掉几乎没声音的片段

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
    """ 使用ffmpeg解码并重采样 """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1", "-ar", str(target_sr),
            "-vn", tmp.name
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {p.stderr.strip()[:300]}")

        audio, sr = sf.read(tmp.name, dtype="float32")
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return audio  # [1, T]


def is_bad_name(fname: str) -> bool:
    return fname.startswith("._") or fname == ".DS_Store"


# =========================
# 3) 多段滑动窗口切分核心逻辑
# =========================
def get_all_segments(waveform: torch.Tensor, target_len: int, hop_size: int) -> list:
    """
    输入: [1, T]
    输出: List[torch.Tensor] 每个形状为 [1, target_len]
    """
    wav = waveform.squeeze(0)
    T = wav.numel()

    # 如果音频短于2秒，做padding返回一段
    if T <= target_len:
        return [F.pad(wav.unsqueeze(0), (0, target_len - T))]

    segments = []
    # 滑动窗口
    for i in range(0, T - target_len + 1, hop_size):
        seg = wav[i: i + target_len]

        # 能量过滤：计算均方根 (RMS)，跳过纯背景噪音
        rms = torch.sqrt((seg ** 2).mean())
        if rms > MIN_RMS:
            segments.append(seg.unsqueeze(0))

    # 如果全文件能量都低，保底取最中间的一段
    if len(segments) == 0:
        mid = (T - target_len) // 2
        segments.append(wav[mid: mid + target_len].unsqueeze(0))

    return segments


# =========================
# 4) 主提取循环
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="调试用，限制处理文件数")
    args = parser.parse_args()

    print("🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    # 读取标签表
    df_label = pd.read_csv(COSWARA_CSV)
    df_label["id"] = df_label["id"].astype(str).str.strip()
    label_lookup = dict(zip(df_label["id"], df_label["covid_status"]))

    # 扫描文件
    audio_tasks = []
    for root, _, files in os.walk(BASE_DIR):
        if os.path.abspath(SAVE_DIR) in os.path.abspath(root): continue
        for f in files:
            if is_bad_name(f): continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in [".wav", ".webm"]: continue

            # 解析 user_id
            parts = root.split(os.sep)
            u_id = next((p for p in parts if len(p) > 20), None)
            if not u_id: continue

            status = label_lookup.get(u_id, None)
            if status not in LABEL_MAP: continue

            audio_tasks.append({
                "user_id": u_id,
                "path": os.path.join(root, f),
                "fname": f,
                "label": LABEL_MAP[status],
                "status": status
            })

    if args.limit > 0: audio_tasks = audio_tasks[:args.limit]
    print(f"📊 待处理音频文件: {len(audio_tasks)} 个")

    rows = []
    stats = {"success_files": 0, "total_segs": 0, "fail": 0}

    with torch.no_grad():
        for task in tqdm(audio_tasks):
            try:
                # 1. 解码
                wav = load_audio_ffmpeg(task["path"], TARGET_SR)

                # 2. 滑窗切分
                hop_samples = int(TARGET_SR * HOP_SEC)
                segments = get_all_segments(wav, TARGET_LEN, hop_samples)

                # 3. 提取特征
                for idx, wav2s in enumerate(segments):
                    # HeAR 提取特征
                    spec = preprocess_audio(wav2s)
                    out = model(spec.to(DEVICE), return_dict=True, output_hidden_states=True)
                    # 取 patch tokens: [1, 97, 1024] -> [96, 1024]
                    feat = out.hidden_states[0][:, 1:, :].squeeze(0).cpu().numpy()

                    # 保存 .npy
                    clean_fname = task['fname'].replace('.', '_')
                    save_name = f"{task['user_id']}_{clean_fname}_seg{idx}.npy"
                    save_path = os.path.join(SAVE_DIR, save_name)
                    np.save(save_path, feat)

                    rows.append({
                        "user_id": task["user_id"],
                        "audio_file": task["fname"],
                        "segment_id": idx,
                        "label": task["label"],
                        "feature_path": save_path,
                        "covid_status": task["status"]
                    })
                    stats["total_segs"] += 1

                stats["success_files"] += 1

            except Exception as e:
                stats["fail"] += 1
                # print(f"Error {task['fname']}: {e}")

    # 保存索引 CSV
    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\n✨ 处理完成！")
        print(f"✅ 成功文件数: {stats['success_files']}")
        print(
            f"📦 生成片段总数: {stats['total_segs']} (平均每个文件 {stats['total_segs'] / stats['success_files']:.1f} 段)")
        print(f"📄 CSV 索引已保存: {OUT_CSV}")
    else:
        print("❌ 未提取到任何有效特征，请检查路径。")


if __name__ == "__main__":
    main()