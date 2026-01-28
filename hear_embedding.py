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
SAVE_DIR = os.path.join(BASE_DIR, "raw_patches_v3")
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


# ================= 核心主程序 =================
def main():
    print(f"🚀 Loading HeAR model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    import hear.python.data_processing.audio_utils as audio_utils
    preprocess_audio = audio_utils.preprocess_audio

    # --- 标签加载逻辑 ---
    diag_map = {}
    diagnosis_path = os.path.join(BASE_DIR, "patient_diagnosis.csv")
    if os.path.exists(diagnosis_path):
        df_diag = pd.read_csv(diagnosis_path, header=None)
        for _, row in df_diag.iterrows():
            diag_map[str(row[0]).strip()] = 0 if str(row[1]).strip().upper() == "HEALTHY" else 1

    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"🚧 正在截获 HeAR 输入端特征 (Exact Match)...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != TARGET_SR:
                    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

                waveform_seg = get_most_informative_segment(waveform, TARGET_LEN)
                spec = preprocess_audio(waveform_seg).to(DEVICE)

                # --- 100% 还原进入 VIT 之前的操作 ---
                # 直接调用 model.embeddings()。
                # 这个方法内部会自动完成：
                # 1. Patch Projection (切片并投影到 1024 维)
                # 2. 加上 CLS Token (增加 1 个长度)
                # 3. 加上 Position Embeddings (位置编码)
                # 这就是 Transformer Encoder 见到的第一个输入。
                x = model.embeddings(spec)

                # 转换为 numpy
                # 形状应该是 (577, 1024) -> 1 个 CLS + 576 个音频补丁
                # 或者根据你的环境可能是 (97, 1024)
                feature_np = x.squeeze(0).cpu().numpy()

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

    print(f"\n✅ 提取完成！")
    print(f"📐 此时生成的特征维度为: {feature_np.shape}")
    print(f"💡 这就是 HeAR 进入第一个 Transformer Block 之前的原始输入。")


if __name__ == "__main__":
    main()