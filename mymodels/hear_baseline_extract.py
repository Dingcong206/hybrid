import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import from_pretrained_keras
from tqdm import tqdm

# --- 1. 官方配置常量 ---
SAMPLE_RATE = 16000  # 官方标准采样率
CLIP_DURATION = 2  # 官方标准片段时长 (秒)
# 对应采样点数：16000 * 2 = 32000
# 模型内部 Reshape 要求的 32240 是由于前端 STFT 补齐产生的，tf.signal.frame 会处理它
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION

# --- 2. 路径设置 ---
# 替换为你自己的 snapshot 路径
local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features_official"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载官方模型 ---
print("📦 正在加载 HeAR 官方模型...")
# 官方加载方式：使用 from_pretrained_keras 直接加载 SavedModel
hear_model = from_pretrained_keras(local_snapshot_path)
hear_infer = hear_model.signatures["serving_default"]


# --- 4. 官方音频预处理函数 ---
def resample_audio_and_convert_to_mono(audio_array, sampling_rate):
    """
    完全参照官方 quick_start notebook 中的重采样与单声道转换逻辑
    """
    # 转换为单声道
    if audio_array.ndim > 1:
        audio_mono = np.mean(audio_array, axis=1)
    else:
        audio_mono = audio_array

    # 使用 librosa 重采样到 16kHz (官方 notebook 使用了 scipy.signal，效果一致但 librosa 更稳)
    if sampling_rate != SAMPLE_RATE:
        audio_mono = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)

    return audio_mono


# --- 5. 批量提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始处理 {len(wav_files)} 个音频文件...")

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    if os.path.exists(save_path): continue

    try:
        # A. 加载原始音频
        audio, sr = librosa.load(file_path, sr=None)

        # B. 官方预处理
        audio_16k = resample_audio_and_convert_to_mono(audio, sr)

        # C. 官方分帧逻辑 (解决 32240/32000 报错的关键)
        # 如果音频不足 2 秒，进行补齐
        if len(audio_16k) < CLIP_LENGTH:
            audio_16k = np.pad(audio_16k, (0, CLIP_LENGTH - len(audio_16k)), mode='constant')

        # 使用 tf.signal.frame 将长音频切成 2 秒一段的 Batch
        # 这就是官方 demo 里的批量推理核心
        # frame_step 设置为 CLIP_LENGTH 表示无重叠切割
        audio_clip_batch = tf.signal.frame(audio_16k, CLIP_LENGTH, CLIP_LENGTH)

        # D. 官方推理
        # 这里的 input 必须是 float32 类型的 constant
        output = hear_infer(x=tf.constant(audio_clip_batch, dtype=tf.float32))

        # E. 提取 Embedding
        # output_0 包含的就是 ViT 提取的高维特征
        embeddings = output['output_0'].numpy()

        # 保存结果 (shape: [片段数量, 512])
        np.save(save_path, embeddings)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！特征保存在: {SAVE_DIR}")