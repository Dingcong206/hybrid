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

        # ... (前面的加载和 16k 转换保持不变) ...

        # B. 补齐与分帧
        # 注意：HeAR 前端通常期望略多于 32000 点（为了 STFT 边缘对齐）
        # 我们按照报错提示的 32240 来补齐
        TARGET_LEN = 32000
        if len(audio_16k) < TARGET_LEN:
            audio_16k = np.pad(audio_16k, (0, TARGET_LEN - len(audio_16k)), mode='constant')

        # 将长音频切成片段 [N, 32240]
        audio_clips = tf.signal.frame(audio_16k, TARGET_LEN, TARGET_LEN)

        # C. 逐个片段提取 Patch (解决 Reshape 报错的关键)
        all_patches = []
        for i in range(audio_clips.shape[0]):
            single_clip = audio_clips[i]  # 形状 [32240]
            # 增加 Batch 维度变为 [1, 32240]
            input_tensor = tf.expand_dims(single_clip, axis=0)

            # 调用前端
            output = patch_infer(audio_wav=input_tensor)
            all_patches.append(output['output_0'].numpy())

        # D. 合并结果
        # 最终形状: [N, 190, 256]
        final_patches = np.concatenate(all_patches, axis=0)

        # E. 保存
        np.save(save_path, final_patches)
    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！真正的 Patch 数据保存在: {SAVE_DIR}")