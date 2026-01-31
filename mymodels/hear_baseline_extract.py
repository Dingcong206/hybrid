import os
import numpy as np
import tensorflow as tf
import librosa
from huggingface_hub import snapshot_download, from_pretrained_keras
from tqdm import tqdm

# 1. 路径配置
DATASET_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUTPUT_DIR = "/data/dingcong/hybrid/features/hear_events"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 下载并加载完整模型组件
print("📦 正在下载 HeAR 完整仓库快照...")
# 这一步会下载整个仓库，包含 encoder, frontend 和 event_detector 子文件夹
local_path = snapshot_download(repo_id="google/hear")

print("🔗 正在从本地路径初始化组件...")
# 基础编码器（位于根目录）
hear_model = from_pretrained_keras(local_path)
# 前端处理与事件检测器（位于子目录）
frontend_model = from_pretrained_keras(os.path.join(local_path, "frontend"))
event_detector = from_pretrained_keras(os.path.join(local_path, "event_detector"))


def extract_events():
    wav_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    print(f"📂 开始处理 {len(wav_files)} 个音频文件...")

    for filename in tqdm(wav_files):
        try:
            audio_path = os.path.join(DATASET_DIR, filename)

            # 预处理：16kHz, 单声道, 2秒长度
            audio, _ = librosa.load(audio_path, sr=16000)
            if len(audio) > 32000:
                audio = audio[:32000]
            else:
                audio = np.pad(audio, (0, 32000 - len(audio)))

            audio_tensor = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

            # --- 核心调用链 ---
            # 1. 前端处理音频
            audio_features = frontend_model(audio_tensor)
            # 2. 生成 Embedding
            embeddings = hear_model.signatures["serving_default"](x=audio_features)['output_0']
            # 3. 真正调用事件检测器！
            predictions = event_detector(embeddings)

            # 保存结果 (通常是各类别的概率分布)
            save_name = filename.replace('.wav', '_events.npy')
            np.save(os.path.join(OUTPUT_DIR, save_name), predictions.numpy())

        except Exception as e:
            print(f"\n❌ 处理 {filename} 失败: {e}")


if __name__ == "__main__":
    extract_events()
    print(f"\n✅ 检测完成！结果已保存至: {OUTPUT_DIR}")