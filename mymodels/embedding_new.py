import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# --- 1. 官方配置常量 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000

# --- 2. 路径设置 (关键修正) ---
# 基础 snapshot 路径
base_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
# 【核心修正】：根据 Notebook 线索，直接加载 frontend 子目录
frontend_model_path = os.path.join(base_snapshot_path, "event_detector", "spectrogram_frontend")

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_patch_final"  # 建议换个目录，存真正的 Patch
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载官方前端模型 ---
print(f"📦 正在加载 HeAR 前端提取器: {frontend_model_path}")
# 使用 tf.saved_model.load 直接加载子模块，这样最稳
patch_model = tf.saved_model.load(frontend_model_path)
patch_infer = patch_model.signatures["serving_default"]


# --- 4. 官方音频预处理函数 ---
def resample_audio_and_convert_to_mono(audio_array, sampling_rate):
    if audio_array.ndim > 1:
        audio_mono = np.mean(audio_array, axis=1)
    else:
        audio_mono = audio_array
    if sampling_rate != SAMPLE_RATE:
        audio_mono = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)
    return audio_mono


# --- 5. 批量提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始处理 {len(wav_files)} 个音频，截取进入 Encoder 之前的 Patch...")

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    if os.path.exists(save_path): continue

    try:
        # A. 加载与预处理
        audio, sr = librosa.load(file_path, sr=None)
        audio_16k = resample_audio_and_convert_to_mono(audio, sr)

        # B. 补齐与分帧
        if len(audio_16k) < CLIP_LENGTH:
            audio_16k = np.pad(audio_16k, (0, CLIP_LENGTH - len(audio_16k)), mode='constant')

        # 将音频切成 2s 的 Batch [N, 32000]
        audio_clip_batch = tf.signal.frame(audio_16k, CLIP_LENGTH, CLIP_LENGTH)

        # C. 官方前端推理 (关键点)
        # 注意：这里的输入 Key 必须是 'audio_wav'
        output = patch_infer(audio_wav=tf.constant(audio_clip_batch, dtype=tf.float32))

        # D. 提取 Patch
        # 此时得到的 patches 维度应该是 (N, 190, 256)
        # N 是 2s 片段的数量，190 是序列长度，256 是 Patch 维度
        patches = output['output_0'].numpy()

        # E. 保存
        np.save(save_path, patches)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！真正的 Patch 数据保存在: {SAVE_DIR}")