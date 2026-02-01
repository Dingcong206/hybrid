import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# --- 1. 官方配置对齐 ---
# 根据官方 audio_utils.py 确定的标准
SAMPLE_RATE = 16000
CLIP_DURATION = 2
FRAME_LENGTH = 32000  # 严格使用 32000，模型内部会处理 STFT 所需的补齐

# --- 2. 路径配置 ---
# 你的 snapshot 真实路径
BASE_PATH = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
# 锁定官方定义的 frontend 子目录
FRONTEND_PATH = os.path.join(BASE_PATH, "event_detector", "spectrogram_frontend")

WAV_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
SAVE_DIR = "/data/dingcong/hybrid/hear_patch_final2"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. 加载官方模型 ---
print(f"📦 正在加载 HeAR 前端提取器: {FRONTEND_PATH}")
# 使用 SavedModel 加载签名，这是最通用的官方方式
patch_model = tf.saved_model.load(FRONTEND_PATH)
patch_infer = patch_model.signatures["serving_default"]


# --- 4. 批量处理函数 ---
def process_single_file(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # 强制补齐/裁剪以对齐 2048 长度 (可选)
    # 或者直接使用滑动窗口，20秒音频配合 160 步长约等于 2000 个 Patch

    # --- 核心修改：将 frame_step 设为 160 (10ms) ---
    NEW_STEP = 160
    audio_clips = tf.signal.frame(audio, FRAME_LENGTH, NEW_STEP).numpy()

    num_frames = audio_clips.shape[0]
    all_patches = []

    for i in range(num_frames):
        single_clip = audio_clips[i:i + 1]
        output_dict = patch_infer(audio_wav=tf.constant(single_clip, dtype=tf.float32))
        patch_data = list(output_dict.values())[0].numpy()
        all_patches.append(patch_data)

    # 结果 shape: [T, 190, 256] -> 其中 T 约为 2000
    return np.concatenate(all_patches, axis=0)

    # C. 逐帧推理 (绕过模型内部 Reshape 限制)
    for i in range(num_frames):
        # 取出单帧 [1, 32000]
        single_clip = audio_clips[i:i + 1]

        # 调用推理接口，Key 必须是 'audio_wav'
        output_dict = patch_infer(audio_wav=tf.constant(single_clip, dtype=tf.float32))

        # 动态获取第一个输出 (解决 'output_0' KeyError)
        # 官方子模型输出通常叫 'output_0' 或 'spectrogram'，这样写最保险
        patch_data = list(output_dict.values())[0].numpy()
        all_patches.append(patch_data)

    # 合并成 [N, 190, 256]
    return np.concatenate(all_patches, axis=0)


# --- 5. 执行主循环 ---
wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
print(f"🎬 开始处理 {len(wav_files)} 个文件，目标维度: (N, 190, 256)")

for filename in tqdm(wav_files):
    save_path = os.path.join(SAVE_DIR, filename.replace('.wav', '.npy'))
    if os.path.exists(save_path): continue

    try:
        final_patches = process_single_file(os.path.join(WAV_DIR, filename))
        np.save(save_path, final_patches)
    except Exception as e:
        print(f"❌ 文件 {filename} 处理失败: {str(e)}")

print(f"✅ 处理完成！真正的 Patch 数据已存入: {SAVE_DIR}")