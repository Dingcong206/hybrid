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
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")
SAVE_DIR = os.path.join(BASE_DIR, "raw_patches_v2")  # 存的是拦截后的原始补丁
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000
TARGET_LEN = 32000  # 2秒

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 工具函数 =================
def get_most_informative_segment(waveform, target_len=32000):
    C, T = waveform.shape
    if T <= target_len:
        return F.pad(waveform, (0, target_len - T))
    energy = waveform.pow(2)
    window_sum = F.avg_pool1d(energy.unsqueeze(0), kernel_size=target_len, stride=1600)
    best_idx = torch.argmax(window_sum).item() * 1600
    start = min(best_idx, T - target_len)
    return waveform[:, start: start + target_len]


# ================= 拦截式提取主程序 =================
def main():
    print(f"🚀 Loading HeAR model to {DEVICE}...")
    # 加载预训练模型
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # 这一步是为了获取 HeAR 内部的音频处理函数
    import hear.python.data_processing.audio_utils as audio_utils
    preprocess_audio = audio_utils.preprocess_audio

    # --- 标签加载 (维持原有 0/1 逻辑) ---
    diag_map = {}
    diagnosis_path = os.path.join(BASE_DIR, "patient_diagnosis.csv")
    if os.path.exists(diagnosis_path):
        df_diag = pd.read_csv(diagnosis_path, header=None)
        for _, row in df_diag.iterrows():
            diag_map[str(row[0]).strip()] = 0 if str(row[1]).strip().upper() == "HEALTHY" else 1

    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"🚧 正在拦截 HeAR 前端特征 (跳过 Transformer Encoder)...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != TARGET_SR:
                    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

                waveform_seg = get_most_informative_segment(waveform, TARGET_LEN)
                spec = preprocess_audio(waveform_seg).to(DEVICE)  # [1, 1, H, W]

                # --- 拦截操作：手动调用模型组件 ---
                # 1. 线性投影切片 (Patch Embedding)
                # 输出形状: [1, 576, 1024] (假设没有 CLS token)
                x = model.patch_embed(spec)

                # 2. 加上位置编码 (Position Embedding)
                # 这一步非常重要，否则你的 Mamba 将失去音频的时间顺序感
                # model.pos_embed 形状通常是 [1, 577, 1024] (包含 1 个 CLS)
                pos_embed = model.pos_embed

                # 我们取除了第一个(CLS)以外的位置编码加到 x 上
                # 这样 x 依然保持 [1, 576, 1024]
                x = x + pos_embed[:, 1:, :]

                # 3. 将这层“未经过 Transformer 污染”的原始特征存下来
                feature_np = x.squeeze(0).cpu().numpy()  # [576, 1024]

                save_filename = filename.replace(".wav", ".npy")
                save_path = os.path.join(SAVE_DIR, save_filename)
                np.save(save_path, feature_np)

                user_id = filename.split('_')[0].strip()
                meta_data.append({
                    "user_id": user_id,
                    "feature_path": save_path,
                    "label": diag_map.get(user_id, 0)
                })
            except Exception as e:
                print(f"⚠️ {filename} 失败: {e}")

    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ 任务完成！特征保存至: {SAVE_DIR}")
    print(f"📏 每个文件的特征维度均为: {feature_np.shape}")


if __name__ == "__main__":
    main()