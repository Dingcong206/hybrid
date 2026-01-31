import os
import sys
import tensorflow as tf
import numpy as np
import librosa
from tqdm import tqdm

# 1. 挂载官方代码路径
sys.path.append("/data/dingcong/hybrid/hear/python")
from data_processing import audio_utils

# 2. 定义模型物理路径
# 这里的路径必须指向包含 saved_model.pb 的文件夹
MODEL_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(MODEL_ROOT, "event_detector/spectrogram_frontend")
ENCODER_PATH = MODEL_ROOT  # Encoder 通常在根目录

# 3. 加载模型组件
print("📦 正在加载 HeAR 模型组件...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]

# 4. 配置输入输出
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# --- 核心处理函数 ---
def extract_official_features(wav_path):
    """
    使用官方预处理逻辑，彻底解决 32240 长度报错
    """
    # 使用官方工具加载音频并重采样至 16kHz
    audio = audio_utils.load_audio(wav_path, sample_rate=16000)

    # 【关键逻辑】官方分帧处理：将长音频切成 32240 长度的片段
    # 这步解决了 Input to reshape is a tensor with 320240 values 的错误
    FRAME_LEN = 32240
    frames = tf.signal.frame(audio, frame_length=FRAME_LEN, frame_step=FRAME_LEN, pad_end=True)

    # 执行模型链路
    # Step A: Frontend (Waveform -> Spectrogram)
    # 此时 frames 的 shape 是 [N, 32240]
    spec_output = frontend(audio=frames)['output_0']

    # Step B: Encoder (Spectrogram -> Embedding)
    embeddings = encoder(x=spec_output)['output_0']

    return embeddings.numpy()


# --- 批量运行循环 ---
print(f"🎬 开始处理 {WAV_DIR} 中的音频文件...")
wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    try:
        # 提取特征
        features = extract_official_features(file_path)

        # 保存为 numpy 文件方便后续训练
        np.save(save_path, features)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！所有特征已保存在: {SAVE_DIR}")