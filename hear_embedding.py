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
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")  # 原始音频位置
SAVE_DIR = os.path.join(BASE_DIR, "spec_npy_v2")  # 特征保存位置
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")  # 记录映射关系

# 确保 HEAR 库能被正确加载 (根据你图片显示的目录结构)
HEAR_PATH = os.path.join(BASE_DIR, "hear")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ================= 参数配置 =================
MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000  # HeAR 要求的采样率
TARGET_LEN = 32000  # HeAR 要求的固定长度 (2秒)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 核心工具函数 =================

def get_most_informative_segment(waveform, target_len=32000):
    C, T = waveform.shape
    if T <= target_len:
        # 确保 pad 后仍然是 [1, target_len]
        return F.pad(waveform, (0, target_len - T))

    energy = waveform.pow(2)
    window_sum = F.avg_pool1d(energy.unsqueeze(0), kernel_size=target_len, stride=1600)
    best_idx = torch.argmax(window_sum).item() * 1600
    start = min(best_idx, T - target_len)

    # 返回 [1, target_len]
    return waveform[:, start: start + target_len]

# ================= 提取主程序 =================
def main():
    print(f"🚀 Loading HeAR model to {DEVICE}...")
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    import hear.python.data_processing.audio_utils as audio_utils
    preprocess_audio = audio_utils.preprocess_audio

    # --- 核心修改：针对 ICBHI 的标签转换逻辑 ---
    diag_map = {}
    diagnosis_path = os.path.join(BASE_DIR, "patient_diagnosis.csv")
    if os.path.exists(diagnosis_path):
        try:
            # 读取 CSV，假设没有表头，第一列是 ID，第二列是诊断文本
            df_diag = pd.read_csv(diagnosis_path, header=None)
            for _, row in df_diag.iterrows():
                p_id = str(row[0]).strip()
                diag_text = str(row[1]).strip().upper()

                # 定义分类逻辑：Healthy 为 0，其他所有病理状态（COPD, URTI, Asthma 等）为 1
                label = 0 if diag_text == "HEALTHY" else 1
                diag_map[p_id] = label

            print(f"✅ 标签转换完成: Healthy -> 0, Others -> 1. 共记录 {len(diag_map)} 个病人")
        except Exception as e:
            print(f"⚠️ 标签表解析失败: {e}")
    # ---------------------------------------

    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"📦 正在处理 {len(wav_files)} 个 ICBHI 音频文件...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != TARGET_SR:
                    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

                # 能量搜索截断 2 秒
                waveform_seg = get_most_informative_segment(waveform, TARGET_LEN)

                # 预处理 (确保 rank 2)
                spec = preprocess_audio(waveform_seg).to(DEVICE)

                # 提取 HeAR Embedding
                patch_embeddings = model.embeddings(spec)
                feature_np = patch_embeddings.squeeze(0).cpu().numpy()

                # 保存 .npy
                save_filename = filename.replace(".wav", ".npy")
                save_path = os.path.join(SAVE_DIR, save_filename)
                np.save(save_path, feature_np)

                # 匹配标签：取文件名前 3 位 ID
                user_id = filename.split('_')[0].strip()
                # 即使文件名是 101，诊断表可能是 101，通过 strip() 确保匹配
                final_label = diag_map.get(user_id, 0)

                meta_data.append({
                    "user_id": user_id,
                    "feature_path": save_path,
                    "label": final_label
                })

            except Exception as e:
                print(f"⚠️ 处理 {filename} 失败: {e}")

    # 保存并打印自检信息
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)

    if not df.empty:
        pos_count = df['label'].sum()
        print(f"\n--- 提取总结 ---")
        print(f"✅ 总样本: {len(df)}")
        print(f"🔥 正样本 (异常): {pos_count} | 负样本 (健康): {len(df) - pos_count}")

        sample_feat = np.load(df.iloc[0]['feature_path'])
        print(f"📐 特征维度: {sample_feat.shape}")
        print(f"💡 请确保 train1.py 中的 input_dim = {sample_feat.shape[1]}")
    else:
        print("❌ 错误：未能生成任何有效特征")


if __name__ == "__main__":
    main()