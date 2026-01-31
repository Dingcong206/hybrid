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
SAVE_DIR = "/data/dingcong/hybrid/hear_features_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 加载模型
print("📦 正在初始化 HeAR ViT Encoder...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]

# --- HeAR baseline 关键常量 ---
SR_TARGET = 16000
CLIP_LEN_2S = SR_TARGET * 2        # 32000
FRAME_LEN_FRONTEND = 32240         # frontend 要求

def pick_loudest_2s(audio_16k_np: np.ndarray) -> np.ndarray:
    """
    HeAR baseline: 对没有时间戳的长音频，只取能量最高的 2 秒。
    能量用 mean(square)（等价于 RMS/dB 的排序）。
    返回 shape: (32000,)
    """
    x = audio_16k_np.astype(np.float32)

    # 不足 2 秒：补 0 到 2 秒
    if len(x) < CLIP_LEN_2S:
        x = np.pad(x, (0, CLIP_LEN_2S - len(x)), mode="constant")
        return x[:CLIP_LEN_2S]

    # 切成不重叠 2 秒段
    frames = tf.signal.frame(x, frame_length=CLIP_LEN_2S, frame_step=CLIP_LEN_2S, pad_end=False)  # [K, 32000]
    # 计算能量
    energy = tf.reduce_mean(tf.square(frames), axis=1)  # [K]
    idx = tf.argmax(energy)
    best = frames[idx]  # [32000]
    return best.numpy()

def pad_32000_to_32240(x_32000: np.ndarray) -> np.ndarray:
    """
    把 2s(32000) 补齐到 frontend 要求的 32240。
    直接尾部补 0（最稳，不引入额外内容）。
    返回 shape: (32240,)
    """
    if x_32000.shape[0] != CLIP_LEN_2S:
        raise ValueError(f"Expected 32000 samples, got {x_32000.shape[0]}")
    if FRAME_LEN_FRONTEND < CLIP_LEN_2S:
        raise ValueError("FRAME_LEN_FRONTEND must be >= 32000")
    if FRAME_LEN_FRONTEND == CLIP_LEN_2S:
        return x_32000
    return np.pad(x_32000, (0, FRAME_LEN_FRONTEND - CLIP_LEN_2S), mode="constant")

def extract_features_hear_baseline(wav_path: str) -> np.ndarray:
    """
    1) load -> 2) resample+mono -> 3) 选能量最大2秒 -> 4) pad到32240 -> 5) frontend -> encoder
    最终返回 shape: (512,)  (每条音频一个 embedding)
    """
    # A. 读音频（float32）
    y, sr = librosa.load(wav_path, sr=None, mono=True, dtype=np.float32)

    # B. 用官方函数重采样/单声道（它期望 (channels, samples)）
    y_torch = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    y_16k = audio_utils.resample_audio_and_convert_to_mono(y_torch, int(sr), SR_TARGET)
    audio_np = y_16k.squeeze().numpy().astype(np.float32)  # (T,)

    # C. 关键：只选 1 个能量最大 2 秒
    clip_2s = pick_loudest_2s(audio_np)  # (32000,)

    # D. 对齐 frontend: 32000 -> 32240
    clip_32240 = pad_32000_to_32240(clip_2s)  # (32240,)

    # E. 送入 frontend/encoder（保持 batch 维： [1, 32240]）
    frames = tf.constant(clip_32240[None, :], dtype=tf.float32)  # (1, 32240)

    spec_output = frontend(audio=frames)["output_0"]
    emb = encoder(x=spec_output)["output_0"]  # 通常 (1,512)

    return emb.numpy().squeeze()  # -> (512,)

# 4. 批量运行
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.lower().endswith(".wav")])
print(f"🎬 开始处理 {len(wav_files)} 个呼吸音文件...")

fail_list = []
for filename in tqdm(wav_files):
    save_path = os.path.join(SAVE_DIR, filename.rsplit(".", 1)[0] + ".npy")
    if os.path.exists(save_path):
        continue

    try:
        feat = extract_features_hear_baseline(os.path.join(WAV_DIR, filename))
        # 每个 wav 保存一个 512 向量，最方便后续训练
        np.save(save_path, feat)
    except Exception as e:
        fail_list.append((filename, str(e)))
        print(f"❌ {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！特征保存在: {SAVE_DIR}")
if fail_list:
    print("\n⚠️ 失败文件（前 20 个）：")
    for item in fail_list[:20]:
        print(item)
