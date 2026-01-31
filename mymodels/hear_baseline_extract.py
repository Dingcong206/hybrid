import os
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import librosa
from tqdm import tqdm  # 进度条库

# 1. 配置路径 (请根据你的实际路径修改)
# 这里的路径应该是你存放 920 个 wav 文件的那个文件夹
DATASET_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUTPUT_DIR = "/data/dingcong/hybrid/features/hear_icbhi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 加载模型 (RTX 4090 加速)
print("🔗 正在加载 HeAR 模型到 GPU...")
model = from_pretrained_keras("google/hear")
infer = model.signatures["serving_default"]


def extract_and_save():
    # 获取文件夹下所有的 .wav 文件
    wav_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    print(f"📂 发现 {len(wav_files)} 个音频文件。开始提取...")

    for filename in tqdm(wav_files):
        try:
            audio_path = os.path.join(DATASET_DIR, filename)

            # 读取音频 (16kHz)
            audio, _ = librosa.load(audio_path, sr=16000)

            # 转换为 Tensor 并增加 batch 维度
            audio_tensor = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

            # 推理 (HeAR 特征提取)
            output = infer(audio_tensor)

            # 提取 embedding (通常是 1024 维)
            feat = output['embedding'].numpy()

            # 保存为 npy 文件，文件名保持一致 (例如 101_1b1.wav -> 101_1b1.npy)
            save_name = filename.replace('.wav', '.npy')
            np.save(os.path.join(OUTPUT_DIR, save_name), feat)

        except Exception as e:
            print(f"❌ 处理 {filename} 出错: {e}")


if __name__ == "__main__":
    extract_and_save()
    print(f"✅ 所有特征已保存至: {OUTPUT_DIR}")