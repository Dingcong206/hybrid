import os
import sys
import torch
import tensorflow as tf
import numpy as np
import librosa
from tqdm import tqdm

# 1. 挂载官方路径
sys.path.append("/data/dingcong/hybrid/hear/python")
from data_processing import audio_utils

# 2. 路径配置
MODEL_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(MODEL_ROOT, "event_detector/spectrogram_frontend")
ENCODER_PATH = MODEL_ROOT

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 加载模型
print("📦 正在初始化 HeAR ViT Encoder...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]


def extract_features_no_error(wav_path):
    # A. 基础加载：强制返回 float32 类型，避免 mean() 报错
    y, sr = librosa.load(wav_path, sr=None, dtype=np.float32)

    # B. 转换为 Torch 并增加 Batch/Channel 维度
    # 官方函数 resample_audio_and_convert_to_mono 期待 (channels, samples) 形状
    y_torch = torch.from_numpy(y).unsqueeze(0)

    # C. 调用官方重采样
    # 显式传递整数类型的采样率，防止 dtype 冲突
    y_16k = audio_utils.resample_audio_and_convert_to_mono(y_torch, int(sr), 16000)
    audio_np = y_16k.squeeze().numpy()

    # D. 【关键：解决 32240 报错】官方规格分帧
    FRAME_LEN = 32240
    frames = tf.signal.frame(audio_np, frame_length=FRAME_LEN, frame_step=FRAME_LEN, pad_end=True)

    # E. 进军 ViT 空间
    # Step 1: Frontend (处理 32240 块)
    spec_output = frontend(audio=frames)['output_0']

    # Step 2: Encoder (Transformer 提取)
    embeddings = encoder(x=spec_output)['output_0']

    return embeddings.numpy()


# 4. 批量运行
print(f"🎬 开始处理 {len(os.listdir(WAV_DIR))} 个呼吸音文件...")
for filename in tqdm(sorted(os.listdir(WAV_DIR))):
    if not filename.endswith('.wav'): continue
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))
    if os.path.exists(save_path): continue

    try:
        feat = extract_features_no_error(os.path.join(WAV_DIR, filename))
        np.save(save_path, feat)
    except Exception as e:
        print(f"❌ {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！特征保存在: {SAVE_DIR}")