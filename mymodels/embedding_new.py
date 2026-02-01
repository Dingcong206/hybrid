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
# 1. 核心配置 (严格对齐你的 SSA-Mamba)
# ============================
SAMPLE_RATE = 16000
FRAME_LEN = 32000  # 必须是 2秒 (32000个采样点)
STEP_SIZE = 160  # 10ms 步长
TARGET_CLIPS = 2048  # 宏观 Token 数
INTERNAL_PATCHES = 16  # 微观 Patch 数 (190 -> 16)
IN_DIM = 48  # HeAR 原始维度
OUT_DIM = 256  # 你的模型需要的维度

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_16x256_final"
os.makedirs(SAVE_DIR, exist_ok=True)

HEAR_BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(HEAR_BASE, "event_detector", "spectrogram_frontend")


# ============================
# 2. 工具组件
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
        # x: [B, T_orig, 48] -> T_orig 通常是 190 或 200
        x = x.permute(0, 2, 1)  # [B, 48, T_orig]
        x = F.interpolate(x, size=self.target_patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [B, 16, 48]
        x = self.proj(x)  # [B, 16, 256]
        return x


# ============================
# 3. 初始化与资源分配
# ============================
print("🚀 正在加载模型并配置 GPU...")
# 防止 TF 占用所有显存，给 PyTorch 留一点空间
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

frontend = tf.saved_model.load(FRONTEND_PATH)
frontend_fn = frontend.signatures["serving_default"]

# 这里的 compressor 负责把 48 变成 256
compressor = PatchCompressor(in_dim=IN_DIM, out_dim=OUT_DIM, target_patches=INTERNAL_PATCHES).cuda().eval()


# ============================
# 4. 核心处理逻辑
# ============================

def process_wav(wav_path):
    # A. 预处理与去噪
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio = butter_bandpass_filter(audio).astype(np.float32)

    # B. 滑动窗口分帧 (找回长音频信息)
    clips = tf.signal.frame(audio, FRAME_LEN, STEP_SIZE).numpy()

    # 解决截图中的 Reshape 错误：确保每一帧都是精确的 32000
    if clips.shape[-1] != FRAME_LEN:
        # 如果最后一帧长度不够，直接舍弃或补齐
        clips = clips[:, :FRAME_LEN]

        # C. 能量筛选 (选出 2048 个精华)
    energies = np.sqrt(np.mean(clips ** 2, axis=1))
    if len(clips) >= TARGET_CLIPS:
        idx = np.argsort(energies)[-TARGET_CLIPS:]
        idx = sorted(idx)
        selected_clips = clips[idx]
    else:
        # 不足 2048 的处理：循环填充
        pad_size = TARGET_CLIPS - len(clips)
        selected_clips = np.concatenate([clips, clips[:pad_size]], axis=0) if len(clips) > 0 else np.zeros(
            (TARGET_CLIPS, FRAME_LEN))

    # D. HeAR 提取 + 维度转换
    final_list = []
    batch_size = 32  # 4096 显存建议设小，24G 显存设 64-128

    for i in range(0, TARGET_CLIPS, batch_size):
        batch_audio = selected_clips[i: i + batch_size]

        # 1. HeAR 提取 (输出维度 [B, 190, 48])
        out = frontend_fn(audio_wav=tf.constant(batch_audio, dtype=tf.float32))
        spec = list(out.values())[0].numpy()

        # 2. 压缩投影 (解决 48 vs 256 的冲突)
        spec_torch = torch.from_numpy(spec).float().cuda()
        with torch.no_grad():
            # 这一步将 [B, 190, 48] 变为 [B, 16, 256]
            tokens = compressor(spec_torch)

        final_list.append(tokens.cpu().numpy())

    # E. 最终拼接 [2048, 16, 256]
    res = np.concatenate(final_list, axis=0)
    # 展平为 SSA-Mamba 需要的长序列 [32768, 256]
    return res.reshape(-1, OUT_DIM)


# ============================
# 5. 执行监控
# ============================
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])
    for f in tqdm(files):
        out_path = os.path.join(SAVE_DIR, f.replace(".wav", ".npy"))
        if os.path.exists(out_path): continue

        try:
            feat = process_wav(os.path.join(WAV_DIR, f))
            np.save(out_path, feat)
        except Exception as e:
            print(f"\n❌ {f} 提取失败: {e}")