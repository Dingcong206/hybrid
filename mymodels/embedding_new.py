import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import from_pretrained_keras
from tqdm import tqdm

# --- 1. 配置 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION

local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
# 更改保存目录，避免覆盖 Embedding 结果
SAVE_DIR = "/data/dingcong/hybrid/hear_patches_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 加载模型并提取 Frontend 算子 ---
print("📦 正在加载 HeAR 模型并拦截 Frontend...")
full_model = from_pretrained_keras(local_snapshot_path)

# 根据官方 SavedModel 结构，我们要调用的通常是负责音频转 Patch 的子层
# 在 SavedModel 中，这些通常被封装在 model 对象下的某些属性中
# 如果直接调用 full_model.signatures["serving_default"]，它会走完全程到 512 维
# 我们需要获取“预处理+分块”后的中间输出
# 如果你希望通过签名调用，通常入口叫 'spectrogram_frontend'
if "spectrogram_frontend" in full_model.signatures:
    patch_extractor = full_model.signatures["spectrogram_frontend"]
    input_key = 'audio'  # 签名模式下通常叫 audio
else:
    # 退而求其次，使用默认签名的内部张量拦截（如果环境允许访问子层）
    print("⚠️ 未找到独立的前端签名，尝试使用默认推理路径的 Patch 拦截...")
    patch_extractor = full_model.signatures["serving_default"]
    input_key = 'x'


# --- 3. 预处理函数 ---
def resample_audio_and_convert_to_mono(audio_array, sampling_rate):
    if audio_array.ndim > 1:
        audio_mono = np.mean(audio_array, axis=1)
    else:
        audio_mono = audio_array
    if sampling_rate != SAMPLE_RATE:
        audio_mono = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)
    return audio_mono


# --- 4. 批量提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始处理 {len(wav_files)} 个音频，提取中间层 Patches...")

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

        # --- 核心修改：获取 Patch 输出 ---
        # 注意：如果你的目的是获取 Patch 给自己的 Encoder，我们需要输出的是中间层数据
        # 如果模型支持提取中间层（hidden_states），通常返回的是 [Batch, Num_Patches, Dim]
        # 在这里我们假设你要的是进入 ViT 前最原始的补丁序列

        # 传入输入张量
        input_data = {input_key: tf.constant(audio_clip_batch, dtype=tf.float32)}
        output = patch_extractor(**input_data)

        # 【重点】如果你想跳过 Google 的 Encoder，你要找的输出 Key 通常不是 'output_0'
        # 请根据诊断打印出的 key 来替换。通常中间层可能叫 'spectrogram' 或 'patches'
        # 如果诊断只有 'output_0'，你需要确认该 SavedModel 是否包含中间层签名
        target_key = 'output_0'  # 默认占位符，需根据诊断调整

        patches = output[target_key].numpy()

        # 保存结果 (供你的自定义 Encoder 作为输入)
        np.save(save_path, patches)

    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ Patch 提取完成！")