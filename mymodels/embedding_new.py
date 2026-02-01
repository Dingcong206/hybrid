import os
import numpy as np
import librosa
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ============================
# 配置
# ============================
SAMPLE_RATE = 16000
FRAME_LEN = 32000
NUM_PATCHES = 16
IN_DIM = 48
OUT_DIM = 256

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_16x256"
os.makedirs(SAVE_DIR, exist_ok=True)

HEAR_BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
FRONTEND_PATH = os.path.join(
    HEAR_BASE,
    "event_detector",
    "spectrogram_frontend"
)

# ============================
# Patch Embedding
# ============================
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=48, out_dim=256, num_patches=16):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 48, 200]
        x = F.interpolate(x, size=self.num_patches, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [B, 16, 48]
        x = self.proj(x)        # [B, 16, 256]
        return x


# ============================
# 初始化
# ============================
print("Loading HeAR frontend...")
frontend = tf.saved_model.load(FRONTEND_PATH)
frontend_fn = frontend.signatures["serving_default"]

patch_embed = PatchEmbedding(
    in_dim=IN_DIM,
    out_dim=OUT_DIM,
    num_patches=NUM_PATCHES
)

# ============================
# 单文件处理
# ============================
def process_wav(wav_path):
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    if len(audio) < FRAME_LEN:
        audio = np.pad(audio, (0, FRAME_LEN - len(audio)))
    audio = audio[:FRAME_LEN]

    # HeAR frontend
    audio_tf = tf.constant(audio[None, :], dtype=tf.float32)
    out = frontend_fn(audio_wav=audio_tf)
    spec = list(out.values())[0].numpy()  # (1, 200, 48)

    # Patch embedding
    spec_torch = torch.from_numpy(spec).float()
    tokens = patch_embed(spec_torch)

    return tokens.squeeze(0).detach().cpu().numpy()
    # (16, 256)


# ============================
# 批量处理
# ============================
if __name__ == "__main__":
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]

    for wav in tqdm(wav_files):
        wav_path = os.path.join(WAV_DIR, wav)
        save_path = os.path.join(SAVE_DIR, wav.replace(".wav", ".npy"))

        if os.path.exists(save_path):
            continue

        try:
            feat = process_wav(wav_path)
            np.save(save_path, feat)
        except Exception as e:
            print(f"❌ {wav} failed:", e)

    print("✅ All features extracted.")
