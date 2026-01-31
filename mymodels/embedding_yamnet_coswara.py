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

# --- YAMNet (TensorFlow Hub) ---
import tensorflow as tf  # noqa: F401
import tensorflow_hub as hub


# =========================
# 0) 默认路径与配置
# =========================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")

SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patch_tokens_yamnet")
OUT_CSV = os.path.join(BASE_DIR, "coswara_hear_patches_yamnet.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000
TARGET_LEN = 32000  # 2 seconds  16k

LABEL_MAP = {
    "healthy": 0,
    "positive_mild": 1,
    "positive_moderate": 1,
    "positive_asymp": 1,
}

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1) 导入本地 HeAR preprocess_audio
# =========================
def import_hear_preprocess():
    """
    依赖： hear 源码在 /data/dingcong/hybrid/hear/...
    通过把 /data/dingcong/hybrid 加到 sys.path 来 import
    """
    sys.path.insert(0, "/data/dingcong/hybrid")
    import importlib
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    return audio_utils.preprocess_audio


preprocess_audio = import_hear_preprocess()


# =========================
# 2) 解码音频：ffmpeg -> wav -> soundfile
# =========================
def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    返回 waveform: torch.float32 [1, T]
    - 强制单声道
    - 强制重采样到 target_sr
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-vn",
            tmp.name
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {p.stderr.strip()[:300]}")

        audio, sr = sf.read(tmp.name, dtype="float32")  # audio: [T] float32
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # [1, T]
        else:
            # 极少情况多通道：做均值
            audio = audio.mean(dim=1, keepdim=True).transpose(0, 1)

        return audio  # [1, T]


def is_bad_name(fname: str) -> bool:
    return fname.startswith("._") or fname == ".DS_Store"


# =========================
# 3) YAMNet：加载、类别索引、2秒窗口选择
# =========================
_YAMNET = None


def load_yamnet():
    global _YAMNET
    if _YAMNET is None:
        _YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
    return _YAMNET


def get_yamnet_class_index(keyword: str, csv_path: str) -> int:
    """
    从 yamnet_class_map.csv 精确找到类名对应的 index
    """
    df = pd.read_csv(csv_path)
    # 先精确匹配
    hits = df[df["display_name"].str.lower() == keyword.lower()]
    if len(hits) == 0:
        # 再模糊匹配
        hits = df[df["display_name"].str.lower().str.contains(keyword.lower())]
    if len(hits) == 0:
        raise ValueError(f"Cannot find class '{keyword}' in {csv_path}")
    return int(hits.iloc[0]["index"])


def yamnet_scores_16k(wav_1d_16k: torch.Tensor) -> np.ndarray:
    """
    输入：torch [T] 16kHz
    输出：scores [frames, 521]
    """
    m = load_yamnet()
    x = wav_1d_16k.detach().cpu().numpy().astype(np.float32)
    scores, embeddings, spec = m(x)
    return scores.numpy()


def pick_best_2s_by_energy(wav_1d: torch.Tensor, target_len: int, hop: int = 1600) -> torch.Tensor:
    """
    fallback：滑窗找 RMS 能量最大的 2 秒
    hop=1600 表示每 0.1s 移动一次（16000Hz）
    """
    T = wav_1d.numel()
    if T <= target_len:
        return F.pad(wav_1d.unsqueeze(0), (0, target_len - T)).squeeze(0)

    best_s, best_i = -1.0, 0
    for i in range(0, T - target_len + 1, hop):
        seg = wav_1d[i:i + target_len]
        s = float((seg ** 2).mean().item())
        if s > best_s:
            best_s, best_i = s, i
    return wav_1d[best_i:best_i + target_len]


def select_best_2s_yamnet(
    waveform: torch.Tensor,
    sr: int,
    target_len: int,
    cough_class_index: int = None,
    hop: int = 1600,
) -> torch.Tensor:
    """
    waveform: [1, T], sr 应该是 16000
    返回: [1, target_len]
    - cough_class_index 提供：选择 cough 概率最高的 2 秒
    - cough_class_index=None：退化为能量最大 2 秒
    - YAMNet 失败：退化为能量最大 2 秒
    """
    assert waveform.ndim == 2 and waveform.shape[0] == 1, "waveform must be [1,T]"
    wav = waveform.squeeze(0)  # [T]

    if wav.numel() <= target_len:
        return F.pad(wav.unsqueeze(0), (0, target_len - wav.numel()))

    # 理论上不会发生（因为 ffmpeg 强制了 16k）
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)

    # 如果没给 cough 类，直接能量选窗
    if cough_class_index is None:
        best = pick_best_2s_by_energy(wav, target_len, hop=hop)
        return best.unsqueeze(0)

    # 用 YAMNet scores 选窗
    try:
        scores = yamnet_scores_16k(wav)  # [frames, 521]
    except Exception:
        best = pick_best_2s_by_energy(wav, target_len, hop=hop)
        return best.unsqueeze(0)

    frames = scores.shape[0]
    duration_s = wav.numel() / 16000.0
    frame_times = np.linspace(0, duration_s, frames, endpoint=False)

    win_s = target_len / 16000.0
    step_s = hop / 16000.0

    best_score, best_start_t = -1.0, 0.0
    max_start = max(0.0, duration_s - win_s)
    t = 0.0
    while t <= max_start + 1e-9:
        mask = (frame_times >= t) & (frame_times < t + win_s)
        if np.any(mask):
            s = float(scores[mask, cough_class_index].mean())
            if s > best_score:
                best_score, best_start_t = s, t
        t += step_s

    start = int(best_start_t * 16000)
    end = start + target_len
    if end > wav.numel():
        start = max(0, wav.numel() - target_len)
        end = start + target_len

    return wav[start:end].unsqueeze(0)


# =========================
# 4) 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用），0 表示全量")
    parser.add_argument("--yamnet_csv", type=str, default="yamnet_class_map.csv", help="YAMNet class map csv 路径")
    parser.add_argument("--only_cough_files", action="store_true",
                        help="仅对文件名含 cough 的音频用 cough 类评分；其他音频用能量选窗")
    parser.add_argument("--hop", type=int, default=1600, help="滑窗步长（采样点），默认1600=0.1秒")
    args = parser.parse_args()

    print("🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    cough_idx = get_yamnet_class_index("Cough", args.yamnet_csv)
    print(f"✅ YAMNet cough class index = {cough_idx}")

    # 读取标签表
    df = pd.read_csv(COSWARA_CSV)
    df["id"] = df["id"].astype(str).str.strip()
    label_lookup = dict(zip(df["id"], df["covid_status"]))

    # 扫描文件
    stats_scan = {
        "skip_dotfiles": 0,
        "skip_non_audio": 0,
        "skip_small": 0,
        "skip_no_uid": 0,
        "skip_no_label": 0
    }

    audio_tasks = []
    for root, _, files in os.walk(BASE_DIR):
        # 避免扫输出目录
        if os.path.abspath(SAVE_DIR) in os.path.abspath(root):
            continue

        for f in files:
            if is_bad_name(f):
                stats_scan["skip_dotfiles"] += 1
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext not in [".wav", ".webm"]:
                stats_scan["skip_non_audio"] += 1
                continue

            full = os.path.join(root, f)

            # 太小的文件很可能是坏文件
            try:
                if os.path.getsize(full) < 1024:
                    stats_scan["skip_small"] += 1
                    continue
            except OSError:
                stats_scan["skip_small"] += 1
                continue

            # 从路径解析 user_id
            parts = root.split(os.sep)
            u_id = next((p for p in parts if len(p) > 20), None)
            if not u_id:
                stats_scan["skip_no_uid"] += 1
                continue

            status = label_lookup.get(u_id, None)
            if status not in LABEL_MAP:
                stats_scan["skip_no_label"] += 1
                continue

            audio_tasks.append({
                "user_id": u_id,
                "path": full,
                "fname": f,
                "label": LABEL_MAP[status],
                "status": status
            })

    print("\n📌 扫描统计：")
    for k, v in stats_scan.items():
        print(f"  {k}: {v}")
    print(f"\n📊 匹配成功：{len(audio_tasks)} 个音频。开始提取 patch...")

    if args.limit and args.limit > 0:
        audio_tasks = audio_tasks[:args.limit]
        print(f"⚠️ 调试模式：只跑前 {args.limit} 条")

    stats_run = {"success": 0, "ffmpeg_fail": 0, "other_fail": 0}
    rows = []

    with torch.no_grad():
        for task in tqdm(audio_tasks):
            try:
                wav = load_audio_ffmpeg(task["path"], TARGET_SR)  # [1, T]

                # 仅对 cough 文件用 cough 评分，否则能量选窗
                use_cough_score = True
                if args.only_cough_files and ("cough" not in task["fname"].lower()):
                    use_cough_score = False

                if use_cough_score:
                    wav2s = select_best_2s_yamnet(
                        wav, sr=TARGET_SR, target_len=TARGET_LEN,
                        cough_class_index=cough_idx, hop=args.hop
                    )
                    window_method = "yamnet_cough"
                else:
                    wav2s = select_best_2s_yamnet(
                        wav, sr=TARGET_SR, target_len=TARGET_LEN,
                        cough_class_index=None, hop=args.hop
                    )
                    window_method = "energy"

                # HeAR preprocess -> spec [1,1,192,128]
                spec = preprocess_audio(wav2s)

                # ViT: hidden_states[0] 是 patch embedding + cls（进入block之前）
                out = model(spec.to(DEVICE), return_dict=True, output_hidden_states=True)
                tokens = out.hidden_states[0]      # [1, 97, 1024]
                patch = tokens[:, 1:, :]           # [1, 96, 1024]
                feat = patch.squeeze(0).cpu().numpy()  # (96, 1024)

                save_name = f"{task['user_id']}_{task['fname'].replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feat)

                rows.append({
                    "user_id": task["user_id"],
                    "audio_file": task["fname"],
                    "audio_path": task["path"],
                    "covid_status": task["status"],
                    "label": task["label"],
                    "feature_path": save_path,
                    "feature_shape": str(feat.shape),
                    "window_method": window_method,
                })
                stats_run["success"] += 1

            except RuntimeError as e:
                if "ffmpeg decode failed" in str(e):
                    stats_run["ffmpeg_fail"] += 1
                else:
                    stats_run["other_fail"] += 1

            except Exception:
                stats_run["other_fail"] += 1

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print("\n✨ 运行总结：")
        print(f"✅ success:     {stats_run['success']}")
        print(f"❌ ffmpeg_fail: {stats_run['ffmpeg_fail']}")
        print(f"❌ other_fail:  {stats_run['other_fail']}")
        print(f"📄 CSV saved -> {OUT_CSV}")
        print(f"📁 NPY saved -> {SAVE_DIR}")
    else:
        print("❌ 没有生成任何特征。请检查 combined_data.csv 和数据路径。")


if __name__ == "__main__":
    main()
