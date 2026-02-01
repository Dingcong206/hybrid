import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from scipy.signal import butter, lfilter

# ============================
# 1. 配置参数 (严格对齐)
# ============================
SAMPLE_RATE = 16000
FRAME_LEN = 32000  # 精确 2 秒
STEP_SIZE = 160  # 10ms 滑动步长
TARGET_CLIPS = 2048  # 宏观 Token 数 (T)
INTERNAL_PATCHES = 16  # 微观 Patch 数 (190 -> 16)
IN_DIM = 48  # HeAR 原始维度
OUT_DIM = 256  # 目标特征维度

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_16x256_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

HEAR_BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(HEAR_BASE, "event_detector", "spectrogram_frontend")


# ============================
# 2. 组件定义
# ============================

def butter_bandpass_filter(data, lowcut=100, highcut=2000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


class PatchCompressor(nn.Module):
    def __init__(self, in_dim=48, out_dim=256, target_patches=16):
        super().__init__()
        self.target_patches = target_patches
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [B, T_orig, 48]
        x = x.permute(0, 2, 1)  # [B, 48, T_orig]
        x = F.interpolate(x, size=self.target_patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [B, 16, 48]
        return self.proj(x)  # [B, 16, 256]


# ============================
# 3. 初始化 (支持双显卡)
# ============================
print("📦 正在初始化 HeAR 与 Compressor...")
# 解决 TF 显存强占问题
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

frontend = tf.saved_model.load(FRONTEND_PATH)
frontend_fn = frontend.signatures["serving_default"]

# 使用 GPU 加速投影
compressor = PatchCompressor(IN_DIM, OUT_DIM, INTERNAL_PATCHES).cuda().eval()


# ============================
# 4. 处理流水线
# ============================

def process_wav(wav_path):
    # Step 1: 去噪加载
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio = butter_bandpass_filter(audio).astype(np.float32)

    # Step 2: 滑动窗口 (确保每一帧长度绝对为 32000)
    # 修复 image_22e134.png 的关键：严格分帧
    clips = tf.signal.frame(audio, FRAME_LEN, STEP_SIZE, pad_end=False).numpy()

    # Step 3: 能量筛选 (选出精华 2048 片段)
    energies = np.sqrt(np.mean(clips ** 2, axis=1))
    if len(clips) >= TARGET_CLIPS:
        idx = np.argsort(energies)[-TARGET_CLIPS:]
        idx = sorted(idx)  # 恢复时间顺序
        selected_clips = clips[idx]
    else:
        # 不足 2048 则进行零填充或循环填充
        pad_size = TARGET_CLIPS - len(clips)
        selected_clips = np.pad(clips, ((0, pad_size), (0, 0)), mode='constant')

    # Step 4: 批量特征提取
    final_features = []
    batch_size = 32  # 适配 4090 显存

    for i in range(0, TARGET_CLIPS, batch_size):
        batch_audio = selected_clips[i: i + batch_size]

        # A. HeAR 提取
        # 强制转换为 tf.float32 确保精度
        out_dict = frontend_fn(audio_wav=tf.constant(batch_audio, dtype=tf.float32))
        spec = list(out_dict.values())[0].numpy()  # [Batch, 190, 48]

        # B. 维度对齐投影
        # 修复 image_23e47e.png 的关键：在此处统一转为 256 维
        spec_torch = torch.from_numpy(spec).float().cuda()
        with torch.no_grad():
            tokens = compressor(spec_torch)  # [Batch, 16, 256]

        final_features.append(tokens.cpu().numpy())

    # Step 5: 合并数据
    # 最终形状: (2048, 16, 256) -> 展平为 (32768, 256)
    res = np.concatenate(final_features, axis=0)
    return res.reshape(-1, OUT_DIM)


# ============================
# 5. 主执行循环
# ============================
if __name__ == "__main__":
    wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])

    for wav in tqdm(wav_files):
        save_path = os.path.join(SAVE_DIR, wav.replace(".wav", ".npy"))
        if os.path.exists(save_path): continue

        try:
            feat = process_wav(os.path.join(WAV_DIR, wav))
            np.save(save_path, feat)
        except Exception as e:
            print(f"\n❌ 处理 {wav} 失败: {str(e)[:100]}")

    print(f"\n✅ 特征提取圆满完成，路径: {SAVE_DIR}")