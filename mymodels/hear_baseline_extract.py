import os
import sys
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 核心修改：链接到你下载的官方源码绝对路径 ---
HEAR_SOURCE_DIR = "/data/dingcong/hybrid/hear/python"
if HEAR_SOURCE_DIR not in sys.path:
    sys.path.append(HEAR_SOURCE_DIR)

# 现在可以安全地导入官方模块了
try:
    from data_processing.audio_utils import preprocess_audio

    print("✅ 成功链接官方 data_processing 模块")
except ImportError:
    print(f"❌ 错误：在 {HEAR_SOURCE_DIR} 没找到源码，请核对路径！")
    sys.exit()

from transformers import AutoModel

# --- 路径配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR_PATH = "/data/dingcong/models/hear_event_detector"  # 你解压权重的地方
CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/combined_data.csv"
SAVE_DIR = "/data/dingcong/hybrid/Coswara-Data/official_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 初始化模型 ---
print("🔗 正在加载官方模型 (Detector & ViT)...")
detector = tf.saved_model.load(DETECTOR_PATH).signatures['serving_default']
hear_vit = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True).to(DEVICE).eval()


def get_official_embedding(wav_path):
    import librosa
    # 1. 官方要求的 16k 采样
    y, _ = librosa.load(wav_path, sr=16000)

    # 2. 运行健康检测器 (Event Detector)
    input_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    det_out = detector(input_tensor)
    probs = det_out['probabilities'].numpy()[:, 0]  # 0位是咳嗽

    # 3. 筛选有效段落 (阈值 0.2)
    valid_secs = np.where(probs > 0.2)[0]
    if len(valid_secs) == 0: valid_secs = [np.argmax(probs)]

    all_embs = []
    for sec in valid_secs:
        start, end = int(sec * 16000), int((sec + 2) * 16000)
        seg = y[start:end]
        if len(seg) < 32000: seg = np.pad(seg, (0, 32000 - len(seg)))

        # 4. 官方预处理 + 全量 ViT 提取 1024 维特征
        spec = preprocess_audio(torch.from_numpy(seg).float().unsqueeze(0))
        with torch.no_grad():
            output = hear_vit(spec.to(DEVICE))
            all_embs.append(output.pooler_output.cpu().numpy())

    # 返回均值作为该音频的 Baseline 特征
    return np.mean(np.concatenate(all_embs, axis=0), axis=0)


# --- 执行提取 ---
df = pd.read_csv(CSV_PATH)
for idx, row in tqdm(df.iterrows(), total=len(df)):
    save_path = os.path.join(SAVE_DIR, f"{row['user_id']}.npy")
    if not os.path.exists(save_path):
        try:
            feat = get_official_embedding(row['path'])
            np.save(save_path, feat)
        except Exception as e:
            print(f"❌ 样本 {row['user_id']} 提取失败: {e}")

print(f"✅ 所有特征已保存至: {SAVE_DIR}")