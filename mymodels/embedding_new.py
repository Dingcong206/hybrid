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
    # A. 加载音频并转为单声道 16kHz
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # B. 按照官方逻辑进行分帧 (解决 num_frames 未定义问题)
    # 如果音频不足 2 秒，进行补齐
    if len(audio) < FRAME_LENGTH:
        audio = np.pad(audio, (0, FRAME_LENGTH - len(audio)), mode='constant')

    # 使用 tf.signal.frame 将长音频切成 2 秒一段的 Batch [N, 32000]
    # 这里 frame_step = FRAME_LENGTH 表示无重叠切割
    # 设定 10ms 步长 (160 采样点) 以达到 2K 序列长度
    NEW_STEP = 160
    audio_clips = tf.signal.frame(audio, FRAME_LENGTH, NEW_STEP).numpy()

    num_frames = audio_clips.shape[0]
    all_patches = []

    # 必须严格一帧一帧喂，防止 Reshape 报错
    for i in range(num_frames):
        # 保持形状为 [1, 32000]
        single_clip = audio_clips[i: i + 1]

        # 推理
        output_dict = patch_infer(audio_wav=tf.constant(single_clip, dtype=tf.float32))
        patch_data = list(output_dict.values())[0].numpy()  # [1, 190, 256]

        # 【重要建议】为了节省硬盘，对 190 维度取平均
        # 这样 920 个文件的 2K 序列才不会把你的硬盘塞满
        patch_reduced = np.mean(patch_data, axis=1)  # 变成 [1, 256]
        all_patches.append(patch_reduced)

    # 最终合并成 [T, 256]
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