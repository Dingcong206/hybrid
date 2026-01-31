import os
import sys
import tensorflow as tf
import numpy as np
import librosa
import torch  # 注意：官方预处理函数使用的是 torch 格式
from tqdm import tqdm

# 1. 挂载路径
sys.path.append("/data/dingcong/hybrid/hear/python")
from data_processing import audio_utils

# 2. 模型路径配置
MODEL_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(MODEL_ROOT, "event_detector/spectrogram_frontend")
ENCODER_PATH = MODEL_ROOT

# 加载模型
print("📦 正在加载 HeAR ViT Encoder (RTX 4090 已就绪)...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]


# 3. 官方标准提取函数
def extract_with_official_utils(wav_path):
    # A. 使用 librosa 加载原始音频
    y, sr = librosa.load(wav_path, sr=16000)

    # B. 调用官方重采样与单声道转换
    # 注意：官方函数可能期待 torch.Tensor
    y_torch = torch.from_numpy(y).float()
    processed_audio = audio_utils.resample_audio_and_convert_to_mono(y_torch, sr, 16000)

    # C. 解决 32240 报错的核心：分帧
    # 这里我们手动对齐官方要求的帧长
    FRAME_LEN = 32240
    frames = tf.signal.frame(processed_audio.numpy(), frame_length=FRAME_LEN, frame_step=FRAME_LEN, pad_end=True)

    # D. 进入 ViT 链路
    # Step 1: Frontend
    spec_output = frontend(audio=frames)['output_0']

    # Step 2: Encoder (ViT 核心)
    # 这一步将频谱图转换为高维 Embedding
    embeddings = encoder(x=spec_output)['output_0']

    return embeddings.numpy()


# 4. 批量处理循环
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features_official"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🎬 开始处理 {len(os.listdir(WAV_DIR))} 个音频文件...")
for filename in tqdm(os.listdir(WAV_DIR)):
    if filename.endswith('.wav'):
        try:
            feat = extract_with_official_utils(os.path.join(WAV_DIR, filename))
            np.save(os.path.join(SAVE_DIR, filename.replace('.wav', '.npy')), feat)
        except Exception as e:
            print(f"❌ {filename} 失败: {e}")

print(f"✅ 特征提取完成，已进入 ViT 空间。")