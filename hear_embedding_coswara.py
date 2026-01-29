import os
import sys
import json
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
# 0) 路径与配置
# =========================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")  # 你已有的标签表
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patch_tokens")
OUT_CSV = os.path.join(BASE_DIR, "coswara_hear_patches.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000
TARGET_LEN = 32000  # 2 seconds @ 16k

# 你的 label 映射（按需要改）
LABEL_MAP = {
    "healthy": 0,
    "positive_mild": 1,
    "positive_moderate": 1,
    "positive_asymp": 1,
}

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 1) 确保能 import 本地 hear 的 preprocess_audio
#    注意：你现在验证成功的方式是 importlib.import_module("hear.python....")
#    要让它可用，必须把 /data/dingcong/hybrid 加入 sys.path
#    （因为 hear 包的根在 /data/dingcong/hybrid/hear/python/ hear/..）
# =========================
sys.path.insert(0, "/data/dingcong/hybrid")  # 关键：让 importlib 找到 hear

import importlib
audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
preprocess_audio = audio_utils.preprocess_audio


# =========================
# 2) 音频解码：ffmpeg -> wav -> soundfile
# =========================
def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> torch.Tensor:
    """返回 waveform: torch.float32 [1, T]，已转单声道、重采样到 target_sr"""
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

        audio, sr = sf.read(tmp.name, dtype="float32")
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # [1, T]
        else:
            audio = audio.mean(dim=1, keepdim=True).transpose(0, 1)  # 兜底（很少用到）

        return audio  # [1, T]


def fix_length(waveform: torch.Tensor, target_len: int = 32000) -> torch.Tensor:
    """waveform: [1, T] -> [1, target_len] (center crop or zero pad)"""
    T = waveform.shape[-1]
    if T == target_len:
        return waveform
    if T > target_len:
        start = (T - target_len) // 2
        return waveform[:, start:start + target_len]
    pad_len = target_len - T
    return F.pad(waveform, (0, pad_len))


def is_bad_name(fname: str) -> bool:
    return fname.startswith("._") or fname == ".DS_Store"


# =========================
# 3) 主流程
# =========================
def main():
    print("🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

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
        # 避免扫到你输出目录
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
            try:
                if os.path.getsize(full) < 1024:
                    stats_scan["skip_small"] += 1
                    continue
            except OSError:
                stats_scan["skip_small"] += 1
                continue

            # 从路径里找 user_id（你之前用 len>20 的规则）
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

    stats_run = {"success": 0, "ffmpeg_fail": 0, "other_fail": 0}
    rows = []

    with torch.no_grad():
        for task in tqdm(audio_tasks):
            try:
                wav = load_audio_ffmpeg(task["path"], TARGET_SR)   # [1, T]
                wav = fix_length(wav, TARGET_LEN)                  # [1, 32000]

                # HeAR 官方 preprocess：输出 spec [1,1,192,128]
                spec = preprocess_audio(wav)

                # 前向：取 hidden_states[0] 作为 “ViT block 之前”的 token
                out = model(spec.to(DEVICE), return_dict=True, output_hidden_states=True)

                tokens = out.hidden_states[0]   # [1, 97, 1024]
                patch = tokens[:, 1:, :]        # 去 CLS -> [1, 96, 1024]

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
                    "feature_shape": str(feat.shape)
                })
                stats_run["success"] += 1

            except RuntimeError as e:
                # ffmpeg decode failed
                if "ffmpeg decode failed" in str(e):
                    stats_run["ffmpeg_fail"] += 1
                else:
                    stats_run["other_fail"] += 1

            except Exception:
                stats_run["other_fail"] += 1

    # 保存 CSV
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
