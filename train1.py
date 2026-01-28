import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel

# ================= 路径配置 =================
BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")
SAVE_DIR = os.path.join(BASE_DIR, "spec_npy_v2")
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")
DIAGNOSIS_CSV = os.path.join(BASE_DIR, "patient_diagnosis.csv")  # 真实标签表

HEAR_PATH = os.path.join(BASE_DIR, "hear")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000
TARGET_LEN = 32000

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 工具函数 =================
def get_most_informative_segment(waveform, target_len=32000):
    """确保返回形状为 [1, 32000]"""
    C, T = waveform.shape
    if T <= target_len:
        return F.pad(waveform, (0, target_len - T))
    energy = waveform.pow(2)
    window_sum = F.avg_pool1d(energy.unsqueeze(0), kernel_size=target_len, stride=1600)
    best_idx = torch.argmax(window_sum).item() * 1600
    start = min(best_idx, T - target_len)
    return waveform[:, start: start + target_len]


# ================= 主程序 =================
def main():
    # 1. 加载模型
    model = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model.to(DEVICE).eval()

    import hear.python.data_processing.audio_utils as audio_utils
    preprocess_audio = audio_utils.preprocess_audio

    # 2. 预加载真实标签 (优化：先读取标签表防止最后才发现没标签)
    if os.path.exists(DIAGNOSIS_CSV):
        df_diag = pd.read_csv(DIAGNOSIS_CSV)
        # 假设列名是 user_id 和 label，请根据实际修改
        diag_map = df_diag.set_index(df_diag.columns[0])[df_diag.columns[1]].to_dict()
        print(f"✅ 已加载诊断标签，共 {len(diag_map)} 个 ID")
    else:
        diag_map = {}
        print("⚠️ 警告：未找到 patient_diagnosis.csv，标签将默认为 0")

    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"📦 开始提取 {len(wav_files)} 个文件的特征...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != TARGET_SR:
                    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

                # 保持 [1, 32000]
                waveform_seg = get_most_informative_segment(waveform, TARGET_LEN)

                # 核心修复：送入 [1, 32000] 而不是 [32000]
                spec = preprocess_audio(waveform_seg).to(DEVICE)

                # 提取 Embedding
                patch_embeddings = model.embeddings(spec)
                feature_np = patch_embeddings.squeeze(0).cpu().numpy()

                # 保存
                save_path = os.path.join(SAVE_DIR, filename.replace(".wav", ".npy"))
                np.save(save_path, feature_np)

                # 关联标签
                user_id = filename.split('_')[0]
                label = diag_map.get(user_id, 0)  # 如果找不到 ID 则默认为 0

                meta_data.append({
                    "user_id": user_id,
                    "feature_path": save_path,
                    "label": int(label)
                })

            except Exception as e:
                print(f"❌ 文件 {filename} 处理失败: {e}")

    # 3. 保存并反馈结果
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)

    if not df.empty:
        sample_feat = np.load(df.iloc[0]['feature_path'])
        print(f"\n🚀 提取成功！")
        print(f"📌 特征维度: {sample_feat.shape} (建议 train1.py 的 input_dim 设为 {sample_feat.shape[1]})")
        print(f"📊 正样本(1)数量: {df['label'].sum()} | 总样本: {len(df)}")
    else:
        print("❌ 提取失败，meta_data 为空")


if __name__ == "__main__":
    main()