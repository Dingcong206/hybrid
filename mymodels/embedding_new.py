import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
from scipy.signal import butter, lfilter
import torch
import torch.nn.functional as F  # 使用 torch 的 interpolate 更加丝滑

# --- 1. 配置对齐 ---
SAMPLE_RATE = 16000
FRAME_LENGTH = 32000
TARGET_LEN = 2048  # 宏观 Token 数 (2s 片段数)
INTERNAL_PATCHES = 16  # 每 2s 片段保留的微观 Patch 数

# --- 2. 路径配置 ---
BASE_PATH = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(BASE_PATH, "event_detector", "spectrogram_frontend")
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_patch_res16_final"  # 建议新目录
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载模型 ---
patch_model = tf.saved_model.load(FRONTEND_PATH)
patch_infer = patch_model.signatures["serving_default"]


def butter_bandpass_filter(data, lowcut=100, highcut=2000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


# --- 4. 核心处理逻辑 ---
def process_single_file(file_path):
    # A. 预处理
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio = butter_bandpass_filter(audio).astype(np.float32)

    if len(audio) < FRAME_LENGTH:
        audio = np.pad(audio, (0, FRAME_LENGTH - len(audio)), mode='constant')

    # B. 分帧与能量筛选
    NEW_STEP = 160
    audio_clips = tf.signal.frame(audio, FRAME_LENGTH, NEW_STEP).numpy()
    energies = np.sqrt(np.mean(audio_clips ** 2, axis=1))

    if audio_clips.shape[0] > TARGET_LEN:
        top_indices = np.argsort(energies)[-TARGET_LEN:]
        top_indices = sorted(top_indices)
        selected_clips = audio_clips[top_indices]
    else:
        selected_clips = audio_clips

    all_frame_features = []

    # C. 逐帧推理 + 空间压缩 (190 -> 16)
    for i in range(selected_clips.shape[0]):
        single_clip = selected_clips[i: i + 1]
        output_dict = patch_infer(audio_wav=tf.constant(single_clip, dtype=tf.float32))

        # 原始 patch_data: [1, 190, 256]
        patch_data = list(output_dict.values())[0].numpy()

        # 使用线性插值将 190 压缩到 16
        # 注意：interpolate 需要 [Batch, Channel, Length] 格式
        feat_tensor = torch.from_numpy(patch_data).permute(0, 2, 1)  # [1, 256, 190]
        feat_resized = F.interpolate(feat_tensor, size=INTERNAL_PATCHES, mode='linear', align_corners=False)
        patch_res16 = feat_resized.permute(0, 2, 1).numpy()  # [1, 16, 256]

        all_frame_features.append(patch_res16)

    # D. 最终拼接
    # 结果形状应该是 [TARGET_LEN, 16, 256] -> 展平为 [TARGET_LEN * 16, 256]
    res = np.concatenate(all_frame_features, axis=0)  # [T_actual, 16, 256]

    # 补齐到固定长度
    current_t = res.shape[0]
    if current_t < TARGET_LEN:
        padding = np.zeros((TARGET_LEN - current_t, INTERNAL_PATCHES, 256), dtype=np.float32)
        res = np.concatenate([res, padding], axis=0)

    # 展平以便 Mamba 直接处理：[32768, 256]
    res_flattened = res.reshape(-1, 256)
    return res_flattened


# --- 5. 执行 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
for filename in tqdm(wav_files):
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))
    if os.path.exists(save_path): continue
    try:
        final_data = process_single_file(os.path.join(WAV_DIR, filename))
        np.save(save_path, final_data)
    except Exception as e:
        print(f"❌ {filename} 失败: {e}")