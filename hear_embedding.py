import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel

# ================= 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")  # 存放 wav 和 txt 的地方
SAVE_DIR = os.path.join(BASE_DIR, "segmented_patches_v1")  # 保存 6898 个特征
OUT_CSV = os.path.join(BASE_DIR, "metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000
FIXED_LEN = 32000  # HeAR 偏好 2 秒输入

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 核心工具：对齐补丁 =================
def process_segment(waveform, sr, start_t, end_t):
    # 1. 裁剪音频
    start_sample = int(start_t * sr)
    end_sample = int(end_t * sr)
    chunk = waveform[:, start_sample:end_sample]

    # 2. 统一重采样到 16k
    if sr != TARGET_SR:
        chunk = torchaudio.functional.resample(chunk, sr, TARGET_SR)

    # 3. 填充或截断至 2 秒 (32000个采样点)
    if chunk.shape[1] < FIXED_LEN:
        chunk = F.pad(chunk, (0, FIXED_LEN - chunk.shape[1]))
    else:
        chunk = chunk[:, :FIXED_LEN]
    return chunk


# ================= 提取主程序 =================
def main():
    print(f"🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    import hear.python.data_processing.audio_utils as audio_utils
    preprocess_audio = audio_utils.preprocess_audio

    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"📦 正在解析标注并提取 6898 个呼吸段...")

    with torch.no_grad():
        for wav_name in tqdm(wav_files):
            # 找到对应的 txt 标注文件
            txt_name = wav_name.replace(".wav", ".txt")
            txt_path = os.path.join(WAV_DIR, txt_name)
            wav_path = os.path.join(WAV_DIR, wav_name)

            if not os.path.exists(txt_path): continue

            # 读取音频
            waveform, sr = torchaudio.load(wav_path)

            # 读取 txt 标注 (Start, End, Crackles, Wheezes)
            try:
                # ICBHI txt 通常是以 tab 分隔
                annotations = pd.read_csv(txt_path, sep='\t', header=None)

                for i, row in annotations.iterrows():
                    start_t, end_t, crackle, wheeze = row[0], row[1], int(row[2]), int(row[3])

                    # 裁剪并预处理
                    chunk = process_segment(waveform, sr, start_t, end_t)
                    spec = preprocess_audio(chunk).to(DEVICE)

                    # --- 拦截操作：100% 还原进入 VIT 前的输入 ---
                    x = model.embeddings(spec)
                    feature_np = x.squeeze(0).cpu().numpy()  # (97, 1024)

                    # 唯一标识符: 原文件名_段索引
                    seg_id = f"{wav_name.replace('.wav', '')}_seg_{i}"
                    save_path = os.path.join(SAVE_DIR, f"{seg_id}.npy")
                    np.save(save_path, feature_np)

                    # 标签逻辑：只要有病(Wheeze 或 Crackle)就记为 1
                    label = 1 if (wheeze == 1 or crackle == 1) else 0

                    meta_data.append({
                        "original_wav": wav_name,
                        "segment_id": seg_id,
                        "feature_path": save_path,
                        "label": label
                    })
            except Exception as e:
                print(f"⚠️ 跳过文件 {wav_name}: {e}")

    # 保存新的元数据
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n--- 提取完成 ---")
    print(f"✅ 成功生成呼吸段特征数: {len(df)}")
    print(f"📊 标签分布: 异常(1): {df['label'].sum()} | 正常(0): {len(df) - df['label'].sum()}")
    print(f"💡 现在你可以用这个 CSV 开始训练了！")


if __name__ == "__main__":
    main()