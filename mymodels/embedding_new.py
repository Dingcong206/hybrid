import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. 配置与路径 ---
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000 个采样点

local_snapshot_path = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_patches_final"  # 存储 Patch 的新目录
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 加载模型并锁定前端 (Frontend) ---
print("📦 正在加载 HeAR 官方模型并锁定前端签名...")
full_model = tf.saved_model.load(local_snapshot_path)

# 核心：必须使用 spectrogram_frontend，这是 HeAR 专门将波形转为 Patch 的入口
if "spectrogram_frontend" in full_model.signatures:
    patch_extractor = full_model.signatures["spectrogram_frontend"]
    print("✅ 成功进入官方前端拦截点")
else:
    raise RuntimeError("未能在模型中找到 spectrogram_frontend 签名！")


# --- 3. 预处理函数 ---
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    # 转单声道
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # 重采样
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    # 长度补齐
    if len(audio) < CLIP_LENGTH:
        audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)), mode='constant')
    return audio


# --- 4. 批量提取循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始处理 {len(wav_files)} 个音频，提取原始 Patch 序列...")

for filename in tqdm(wav_files):
    file_path = os.path.join(WAV_DIR, filename)
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))

    if os.path.exists(save_path): continue

    try:
        # A. 准备数据
        audio_16k = preprocess_audio(file_path)
        # 将长音频切成 2s 的 Batch [N, 32000]
        audio_clip_batch = tf.signal.frame(audio_16k, CLIP_LENGTH, CLIP_LENGTH)

        # B. 调用 HeAR 官方前端逻辑
        # 注意：这里的输入 Key 必须叫 audio_wav (由之前的报错得知)
        outputs = patch_extractor(audio_wav=tf.constant(audio_clip_batch, dtype=tf.float32))

        # C. 提取 Patch
        # 此时得到的 patches 维度通常是 (N, 190, 256)
        # 190 是时间序列长度，256 是每个 patch 的特征维数
        patches = outputs['output_0'].numpy()

        # D. 自动检测（确保不是 Embedding）
        if len(patches.shape) < 3 or patches.shape[-1] == 512:
            print(f"⚠️ 警告：{filename} 提取的似乎仍是 Embedding，形状为 {patches.shape}")

        # E. 保存
        np.save(save_path, patches)

    except Exception as e:
        print(f"❌ 处理 {filename} 时发生错误: {e}")

print(f"✅ 所有 Patch 提取完成！存放在: {SAVE_DIR}")


# --- 5. 验证：可视化最后一个处理的文件 ---
def verify_visual(data):
    plt.figure(figsize=(10, 4))
    # 取第一段 Patch 序列，转置显示
    sample = data[0]
    plt.imshow(sample.T, aspect='auto', origin='lower', cmap='magma')
    plt.title("HeAR Native Patches Visualization")
    plt.xlabel("Sequence (Time Steps)")
    plt.ylabel("Patch Dimensions")
    plt.colorbar()
    plt.show()


print("🔍 正在生成最后一个文件的可视化验证...")
verify_visual(patches)