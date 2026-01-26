import os
import librosa
import numpy as np
from tqdm import tqdm

# 配置路径
WAV_DIR = r"D:\Python_project\HeAR\ICBHI_final_database\wav_files"  # 你的wav路径
SAVE_DIR = r"D:\Python_project\HeAR\ICBHI_final_database\spec_npy_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

SR = 16000  # 统一采样率
DURATION = 8  # 统一长度（秒），ICBHI很多音频在5-20s，取8s平衡大部分
TARGET_LEN = 1024  # 时间轴目标长度


def preprocess_audio(file_path):
    # 1. 加载音频
    y, sr = librosa.load(file_path, sr=SR)
    # 2. 填充或裁剪到固定长度
    target_samples = SR * DURATION
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    # 3. 提取Mel频谱
    # n_fft和hop_length决定了时间轴精度
    # hop_length = (SR * DURATION) / TARGET_LEN
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=125)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # 4. 归一化到 0-1 或 -1到1
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)
    return spec_db  # 形状应该接近 (128, 1025)，切一下到1024


# 执行转换
for f in tqdm(os.listdir(WAV_DIR)):
    if f.endswith('.wav'):
        path = os.path.join(WAV_DIR, f)
        spec = preprocess_audio(path)
        np.save(os.path.join(SAVE_DIR, f.replace('.wav', '.npy')), spec[:, :TARGET_LEN])