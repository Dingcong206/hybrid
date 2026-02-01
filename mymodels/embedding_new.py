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
# 配置
# ============================
SAMPLE_RATE = 16000
FRAME_LEN = 32000           # 2s
STEP_SIZE = 160             # 10ms
TARGET_CLIPS = 2048
INTERNAL_PATCHES = 16
IN_DIM = 48
OUT_DIM = 256

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_16x256_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

HEAR_BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(HEAR_BASE, "event_detector", "spectrogram_frontend")

# ============================
# 滤波（可选）
# ============================
def butter_bandpass_filter(data, lowcut=100, highcut=2000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# ============================
# Patch Compressor
# ============================
class PatchCompressor(nn.Module):
    def __init__(self, in_dim=48, out_dim=256, patches=16):
        super().__init__()
        self.patches = patches
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [B, 200, 48]
        x = x.permute(0, 2, 1)                # [B, 48, 200]
        x = F.interpolate(x, size=self.patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)                # [B, 16, 48]
        return self.proj(x)                   # [B, 16, 256]

# ============================
# 初始化模型
# ============================
print("📦 Loading HeAR frontend...")

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frontend = tf.saved_model.load(FRONTEND_PATH)
frontend_fn = frontend.signatures["serving_default"]

device = "cuda" if torch.cuda.is_available() else "cpu"
compressor = PatchCompressor(IN_DIM, OUT_DIM, INTERNAL_PATCHES).to(device).eval()

# ============================
# 主处理函数
# ============================
def process_wav(wav_path):
    # ---- 1. 读取音频 ----
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio = np.nan_to_num(audio).astype(np.float32)

    if len(audio) < FRAME_LEN:
        audio = np.pad(audio, (0, FRAME_LEN - len(audio)))

    # ---- 2. 滑窗 ----
    clips = tf.signal.frame(audio, FRAME_LEN, STEP_SIZE, pad_end=False).numpy()

    if clips.shape[0] == 0:
        clips = np.zeros((1, FRAME_LEN), dtype=np.float32)

    # ---- 3. 能量筛选（重点）----
    energies = np.sqrt(np.mean(clips ** 2, axis=1))

    if clips.shape[0] >= TARGET_CLIPS:
        idx = np.argpartition(energies, -TARGET_CLIPS)[-TARGET_CLIPS:]
        idx = np.sort(idx)  # ✅ 保持时间顺序
        selected = clips[idx]
    else:
        pad = TARGET_CLIPS - clips.shape[0]
        selected = np.pad(clips, ((0, pad), (0, 0)))

    # ---- 4. HeAR frontend（逐条）----
    specs = np.zeros((TARGET_CLIPS, 200, 48), dtype=np.float32)

    for i in range(TARGET_CLIPS):
        clip = selected[i].reshape(1, FRAME_LEN)
        out = frontend_fn(audio_wav=tf.constant(clip, dtype=tf.float32))
        specs[i] = list(out.values())[0][0].numpy()

    # ---- 5. Patch 压缩 ----
    outputs = np.zeros((TARGET_CLIPS, INTERNAL_PATCHES, OUT_DIM), dtype=np.float32)

    batch = 64
    for i in range(0, TARGET_CLIPS, batch):
        x = torch.from_numpy(specs[i:i+batch]).float().to(device)
        with torch.no_grad():
            y = compressor(x)
        outputs[i:i+batch] = y.cpu().numpy()

    # ---- 6. 展平 ----
    return outputs.reshape(-1, OUT_DIM)  # (32768, 256)

# ============================
# 批处理
# ============================
if __name__ == "__main__":
    wavs = sorted([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])

    for wav in tqdm(wavs):
        save_path = os.path.join(SAVE_DIR, wav.replace(".wav", ".npy"))
        if os.path.exists(save_path):
            continue

        try:
            feat = process_wav(os.path.join(WAV_DIR, wav))
            np.save(save_path, feat)
        except Exception as e:
            print(f"❌ {wav} failed:", e)

    print("✅ 所有特征已完成")
