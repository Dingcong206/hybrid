import os
import sys
import torch
import tensorflow as tf
import numpy as np
import librosa
from tqdm import tqdm

# 1. 环境与官方路径挂载
sys.path.append("/data/dingcong/hybrid/hear/python")
from data_processing import audio_utils  # 引入你抓到的官方工具包

# 2. 路径配置
MODEL_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(MODEL_ROOT, "event_detector/spectrogram_frontend")
ENCODER_PATH = MODEL_ROOT

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 加载模型组件 (RTX 4090 并行推理准备)
print("📦 正在初始化 HeAR ViT Encoder...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]


# 4. 官方标准提取流程函数
def extract_official_features(wav_path):
    # A. 基础加载 (保持原始 sr 以便官方函数重采样)
    y, sr = librosa.load(wav_path, sr=None)

    # B. 官方重采样逻辑 (对应 audio_utils.resample_audio_and_convert_to_mono)
    y_torch = torch.from_numpy(y).float().unsqueeze(0)
    # 强制转为 16000Hz 单声道，这是官方 _compute_stft 的前提
    y_16k = audio_utils.resample_audio_and_convert_to_mono(y_torch, sr, 16000)
    audio_np = y_16k.squeeze().numpy()

    # C. 【核心：官方规格分帧】解决 32240 报错
    # 为了适配 ViT 的 190 帧频谱图输入，必须切成 32240 长度
    FRAME_LEN = 32240
    # pad_end=True 会在音频末尾自动补 0，确保每个分段都是 32240
    frames = tf.signal.frame(audio_np, frame_length=FRAME_LEN, frame_step=FRAME_LEN, pad_end=True)

    # D. 深度提取链路 (进入 ViT)
    # Step 1: Frontend (执行音频到频谱的转换，内部对齐 _compute_stft 参数)
    spec_output = frontend(audio=frames)['output_0']

    # Step 2: Encoder (真正的 ViT Transformer 层，进行 Self-Attention 建模)
    # 输出 shape 通常是 [N_segments, Embedding_dim]
    embeddings = encoder(x=spec_output)['output_0']

    return embeddings.numpy()


# 5. 批量执行逻辑
print(f"🎬 开始处理 {WAV_DIR} 下的 920 个呼吸音文件...")
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])

for filename in tqdm(wav_files):
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    # 跳过已处理文件
    if os.path.exists(save_path):
        continue

    try:
        # 执行官方标准提取
        features = extract_official_features(os.path.join(WAV_DIR, filename))

        # 保存特征 (Numpy 格式方便后续训练分类器)
        np.save(save_path, features)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 所有特征提取完成！存储位置: {SAVE_DIR}")