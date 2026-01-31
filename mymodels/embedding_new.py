import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import from_pretrained_keras
from tqdm import tqdm

# --- 1. 配置 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION # 32000

# 路径设置
local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
# 注意：Frontend 通常位于子目录中，或者作为 SavedModel 的一个特定签名
# 根据官方结构，前端通常在 spectrogram_frontend 目录下
frontend_model_path = os.path.join(local_snapshot_path, "event_detector/spectrogram_frontend")

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_patches_output" # 更改保存目录名
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 加载 Frontend 模型 ---
print("📦 正在加载 HeAR Frontend (频谱 Patch 提取器)...")
frontend_model = tf.saved_model.load(frontend_model_path)
frontend_infer = frontend_model.signatures["serving_default"]

def resample_audio_and_convert_to_mono(audio_array, sampling_rate):
    if audio_array.ndim > 1:
        audio_mono = np.mean(audio_array, axis=1)
    else:
        audio_mono = audio_array
    if sampling_rate != SAMPLE_RATE:
        audio_mono = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)
    return audio_mono

# --- 3. 提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始提取 Patch 数据...")

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    if os.path.exists(save_path): continue

    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio_16k = resample_audio_and_convert_to_mono(audio, sr)

        if len(audio_16k) < CLIP_LENGTH:
            audio_16k = np.pad(audio_16k, (0, CLIP_LENGTH - len(audio_16k)), mode='constant')

        # 切分 Batch [N, 32000]
        audio_clip_batch = tf.signal.frame(audio_16k, CLIP_LENGTH, CLIP_LENGTH)

        # --- 核心修改：只运行到 Frontend ---
        # Frontend 的输入 key 通常是 'audio' (根据 SavedModel 签名确认)
        # 它会处理 32000 -> 32240 的补齐并输出 Patch
        patch_output = frontend_infer(audio=tf.constant(audio_clip_batch, dtype=tf.float32))

        # 获取 Patch 数据 (通常 key 是 'output_0')
        # 预期的 Shape 应该是 [N, 190, 256] 左右 (取决于 Patch 大小)
        patches = patch_output['output_0'].numpy()

        # 保存 Patch，供你自己的 Encoder 使用
        np.save(save_path, patches)

    except Exception as e:
        print(f"❌ {filename} 失败: {str(e)}")

print(f"✅ Patch 提取完成！保存在: {SAVE_DIR}")