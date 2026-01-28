import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

# ================= 配置区 =================

BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")  # 原始音频位置
SAVE_DIR = os.path.join(BASE_DIR, "spec_npy")  # 特征保存位置
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")  # 记录映射关系

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

HEAR_PATH = "/data/dingcong/hybrid/hear"
if HEAR_PATH not in sys.path:
    sys.path.append(HEAR_PATH)

# 2. 现在导入就不会报错了

# ================= 提取函数 =================
def main():
    print(f"🚀 Loading HeAR model to {DEVICE}...")
    # 加载预训练模型
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # 获取音频处理工具（由模型库提供）
    import importlib
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    preprocess_audio = audio_utils.preprocess_audio

    # 扫描 WAV 文件
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"📦 Found {len(wav_files)} files. Starting Patch Embedding extraction...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)

            # 1. 加载音频 (假设采样率已由 preprocess_audio 处理或模型内部自适应)
            waveform, sr = torchaudio.load(wav_path)

            # 2. 预处理为 Spectrogram
            # 注意：batch_clips 需要是 [Batch, Time]
            spec = preprocess_audio(waveform).to(DEVICE)

            # 3. 【核心截断步骤】：只通过 Embedding 层
            # 在 google/hear-pytorch 中，model.embeddings 负责 Patchify 和 Positional Encoding
            # 输出形状通常为 [Batch, Seq_Len, Hidden_Dim] (例如 [1, 512, 768])
            patch_embeddings = model.embeddings(spec)

            # 转为 numpy
            feature_np = patch_embeddings.squeeze(0).cpu().numpy()

            # 4. 保存文件
            save_filename = filename.replace(".wav", ".npy")
            save_path = os.path.join(SAVE_DIR, save_filename)
            np.save(save_path, feature_np)

            # 5. 记录元数据 (这里假设标签可以从文件名解析，或后续合并)
            meta_data.append({
                "user_id": filename.split('_')[0],
                "feature_path": save_path,
                "label": 0  # 需根据你的 CSV 逻辑填入真实标签
            })

    # 保存新的 CSV
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Extraction complete. Features saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()