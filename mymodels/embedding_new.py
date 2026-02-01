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
# 1. 配置参数
# ============================
SAMPLE_RATE = 16000
FRAME_LEN = 32000  # 2秒窗口
STEP_SIZE = 160  # 10ms 步长，用于高密度扫描
TARGET_CLIPS = 2048  # 最终保留的宏观片段数
INTERNAL_PATCHES = 16  # 每个2秒片段保留的微观 Token 数
HEAR_DIM = 256  # 最终输出特征维度

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_16x256_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# HeAR 官方模型路径
HEAR_BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(HEAR_BASE, "event_detector", "spectrogram_frontend")


# ============================
# 2. 工具类与函数
# ============================

# --- 呼吸音去噪滤波器 ---
def butter_bandpass_filter(data, lowcut=100, highcut=2000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


# --- Patch 压缩层 (190/200 -> 16) ---
class PatchCompressor(nn.Module):
    def __init__(self, in_dim=48, out_dim=256, target_patches=16):
        super().__init__()
        self.target_patches = target_patches
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x shape: [B, T_orig, 48]
        x = x.permute(0, 2, 1)  # [B, 48, T_orig]
        # 线性插值压缩时间轴
        x = F.interpolate(x, size=self.target_patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [B, 16, 48]
        return self.proj(x)  # [B, 16, 256]


# ============================
# 3. 初始化模型
# ============================
print("📦 正在加载 HeAR 前端 (TF)...")
frontend = tf.saved_model.load(FRONTEND_PATH)
frontend_fn = frontend.signatures["serving_default"]

compressor = PatchCompressor(in_dim=48, out_dim=256, target_patches=INTERNAL_PATCHES).eval()


# ============================
# 4. 核心处理流水线
# ============================
def process_single_wav(wav_path):
    # A. 加载与去噪
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio = butter_bandpass_filter(audio).astype(np.float32)

    # B. 滑动窗口分帧
    # 这一步找回了你丢失的长音频内容
    clips = tf.signal.frame(audio, FRAME_LEN, STEP_SIZE).numpy()

    # C. 能量显著性筛选 (找回 2048)
    energies = np.sqrt(np.mean(clips ** 2, axis=1))
    if len(clips) > TARGET_CLIPS:
        # 选取能量最高的索引并保持时间顺序
        idx = np.argsort(energies)[-TARGET_CLIPS:]
        idx = sorted(idx)
        selected_clips = clips[idx]
    else:
        # 不足 2048 则补齐
        pad_size = TARGET_CLIPS - len(clips)
        selected_clips = np.pad(clips, ((0, pad_size), (0, 0)))

    # D. HeAR 批量特征提取
    final_tokens = []
    batch_size = 64  # 根据显存调整

    with torch.no_grad():
        for i in range(0, TARGET_CLIPS, batch_size):
            batch = selected_clips[i: i + batch_size]

            # 1. HeAR 提取 (TF)
            out = frontend_fn(audio_wav=tf.constant(batch, dtype=tf.float32))
            spec = list(out.values())[0].numpy()  # [batch, 200, 48]

            # 2. 压缩与投影 (Torch)
            spec_t = torch.from_numpy(spec).float()
            tokens = compressor(spec_t)  # [batch, 16, 256]
            final_tokens.append(tokens.cpu().numpy())

    # E. 整理维度 [2048, 16, 256] -> [32768, 256]
    res = np.concatenate(final_tokens, axis=0)
    return res.reshape(-1, HEAR_DIM)


# ============================
# 5. 主循环
# ============================
if __name__ == "__main__":
    wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])

    for wav in tqdm(wav_files):
        save_path = os.path.join(SAVE_DIR, wav.replace(".wav", ".npy"))
        if os.path.exists(save_path): continue

        try:
            feature_matrix = process_single_wav(os.path.join(WAV_DIR, wav))
            np.save(save_path, feature_matrix)
        except Exception as e:
            print(f"❌ 处理 {wav} 失败: {e}")

    print(f"✅ 提取完成！数据保存在: {SAVE_DIR}")