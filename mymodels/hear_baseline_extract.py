import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import from_pretrained_keras
from tqdm import tqdm

# --- 1. 官方配置常量 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000

# --- 2. 路径设置 ---
local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_features_official_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载官方模型 ---
print("📦 正在加载 HeAR 官方模型...")
hear_model = from_pretrained_keras(local_snapshot_path)
hear_infer = hear_model.signatures["serving_default"]

# --- 4. 官方音频预处理函数 ---
def resample_audio_and_convert_to_mono(audio_array, sampling_rate):
    # 转换为单声道
    if audio_array.ndim > 1:
        # librosa.load 默认 mono=True；这里兼容多声道情况
        audio_mono = np.mean(audio_array, axis=0)
    else:
        audio_mono = audio_array

    # 重采样到 16k
    if sampling_rate != SAMPLE_RATE:
        audio_mono = librosa.resample(audio_mono, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)

    return audio_mono.astype(np.float32)

def pick_loudest_2s(audio_16k: np.ndarray) -> np.ndarray:
    """
    HeAR 测试/评测长音频时的策略：
    - 对没有时间戳的长音频，只取能量最高的 2 秒（loudness peak）
    - 能量用 mean(square)（等价于 RMS/dB 的排序）
    返回 shape: (32000,)
    """
    x = audio_16k.astype(np.float32)

    # 不足 2 秒：补 0
    if len(x) < CLIP_LENGTH:
        x = np.pad(x, (0, CLIP_LENGTH - len(x)), mode="constant")
        return x[:CLIP_LENGTH]

    # 切成不重叠 2 秒段（不 pad_end，避免人为引入很多静音段）
    frames = tf.signal.frame(x, frame_length=CLIP_LENGTH, frame_step=CLIP_LENGTH, pad_end=False)  # [K, 32000]

    # 极端兜底：万一 frames 为空（理论上不会）
    if frames.shape[0] == 0:
        return x[:CLIP_LENGTH]

    # 计算每段能量
    energy = tf.reduce_mean(tf.square(frames), axis=1)  # [K]
    idx = tf.argmax(energy)
    best = frames[idx]  # [32000]
    return best.numpy()

# --- 5. 批量提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.lower().endswith(".wav")])
print(f"🎬 开始处理 {len(wav_files)} 个音频文件...")

fail_list = []

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace(".wav", ".npy"))

    if os.path.exists(save_path):
        continue

    try:
        # A. 加载原始音频（建议 mono=True，避免多通道 axis 搞错）
        audio, sr = librosa.load(file_path, sr=None, mono=True)

        # B. 官方预处理：mono + resample 到 16k
        audio_16k = resample_audio_and_convert_to_mono(audio, sr)

        # C. ✅ HeAR 测试策略：只选能量最高的 2 秒
        clip_2s = pick_loudest_2s(audio_16k)  # (32000,)

        # D. 官方推理：保持 batch 维度 [1, 32000]
        output = hear_infer(x=tf.constant(clip_2s[None, :], dtype=tf.float32))

        # E. 提取 embedding：得到 (512,)
        emb = output["output_0"].numpy().squeeze()

        # 保存结果 (shape: [512])
        np.save(save_path, emb)

    except Exception as e:
        fail_list.append((filename, str(e)))
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！特征保存在: {SAVE_DIR}")
if fail_list:
    print("\n⚠️ 失败文件（前 20 个）：")
    for x in fail_list[:20]:
        print(x)
