import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import from_pretrained_keras
from tqdm import tqdm

# --- 1. 官方配置常量 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION

# --- 2. 路径设置 ---
local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
# 专门存放 Patch，不要覆盖 Embedding
SAVE_DIR = "/data/dingcong/hybrid/hear_patches_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载官方模型 (修改处：锁定 Frontend) ---
print("📦 正在加载 HeAR 官方模型前端...")
hear_model = from_pretrained_keras(local_snapshot_path)

# 【核心修改点】
# 原脚本是 hear_model.signatures["serving_default"] (输出 512维)
# 现脚本是 hear_model.signatures["spectrogram_frontend"] (输出 Patches)
patch_infer = hear_model.signatures["spectrogram_frontend"]


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
print(f"🎬 开始提取 Patch 数据 (Encoder 前置特征)...")

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    if os.path.exists(save_path): continue

    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio_16k = resample_audio_and_convert_to_mono(audio, sr)

        if len(audio_16k) < CLIP_LENGTH:
            audio_16k = np.pad(audio_16k, (0, CLIP_LENGTH - len(audio_16k)), mode='constant')

        audio_clip_batch = tf.signal.frame(audio_16k, CLIP_LENGTH, CLIP_LENGTH)

        # --- 6. 官方前端推理 (修改处：Input Key 和 Output) ---
        # 根据你之前的报错，spectrogram_frontend 的输入 Key 是 'audio_wav'
        # 它的输出才是真正进入 VIT Encoder 之前的 Patch 序列
        output = patch_infer(audio_wav=tf.constant(audio_clip_batch, dtype=tf.float32))

        # 这里的 output_0 形状预期为 [N, 190, 256] 左右
        patches = output['output_0'].numpy()

        # 保存结果
        np.save(save_path, patches)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ Patch 提取完成！特征保存在: {SAVE_DIR}")