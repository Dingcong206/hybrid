import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# --- 1. 严格配置模型路径 (根据你的 image_d7d3c6.png 定制) ---
# 这是你 snapshots 文件夹的绝对路径
MODEL_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

# 根据 ls 结果，子模型文件夹名称略有不同
FRONTEND_PATH = os.path.join(MODEL_ROOT, "spectrogram_frontend")
ENCODER_PATH = MODEL_ROOT  # 根目录通常是 Encoder
# 官方提供 large 和 small 两个检测器，这里默认用 large
DETECTOR_PATH = os.path.join(MODEL_ROOT, "event_detector_large")

# 数据路径
DATASET_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUTPUT_DIR = "/data/dingcong/hybrid/features/icbhi_hear_official"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 加载官方组件 ---
print("🔗 正在从本地缓存加载 HeAR 官方组件...")
frontend_model = tf.saved_model.load(FRONTEND_PATH)
hear_encoder = tf.saved_model.load(ENCODER_PATH)
event_detector = tf.saved_model.load(DETECTOR_PATH)

# 获取推理接口
frontend_func = frontend_model.signatures["serving_default"]
encoder_func = hear_encoder.signatures["serving_default"]
detector_func = event_detector.signatures["serving_default"]


# --- 3. 官方复刻：检测 -> 裁剪 2s -> 预测 ---
def process_icbhi_sample(audio_path):
    # A. 采样率重采样至 16kHz (HeAR 官方标准)
    audio, _ = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # B. 寻找最显著的 2 秒 (Event Detection)
    # 链路：Frontend -> Encoder -> Detector
    spec = list(frontend_func(audio=waveform).values())[0]
    emb = list(encoder_func(x=spec).values())[0]
    logits = list(detector_func(x=emb).values())[0]  # [1, Time, Classes]

    # 找到整段音频中异常得分最高的时刻
    event_scores = tf.reduce_max(logits, axis=-1).numpy()[0]
    best_step_idx = np.argmax(event_scores)

    # 将时间步换算回音频采样点，并截取以其为中心的 2 秒 (32000点)
    center_sample = int((best_step_idx / len(event_scores)) * len(audio))
    start = max(0, center_sample - 16000)
    end = min(len(audio), start + 32000)

    # 官方截取并补齐
    crop = audio[start:end]
    if len(crop) < 32000:
        crop = np.pad(crop, (0, 32000 - len(crop)))

    # C. 对截取的 2s 片段做最终二分类
    crop_tensor = tf.convert_to_tensor(crop[np.newaxis, :], dtype=tf.float32)
    final_spec = list(frontend_func(audio=crop_tensor).values())[0]
    final_emb = list(encoder_func(x=final_spec).values())[0]
    final_logits = list(detector_func(x=final_emb).values())[0]

    # 最终异常得分 (0-1)
    # 取该 2s 片段内各类别概率的最大值作为异常强度
    score = np.max(tf.nn.sigmoid(final_logits).numpy())
    prediction = 1 if score > 0.5 else 0

    return prediction, score


# --- 4. 批量执行 ---
def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    results = []

    print(f"🚀 开始处理 {len(files)} 个 ICBHI 样本...")
    with tf.device('/GPU:0'):
        for filename in tqdm(files):
            try:
                path = os.path.join(DATASET_DIR, filename)
                pred, score = process_icbhi_sample(path)
                results.append(f"{filename},{pred},{score:.4f}")
            except Exception as e:
                print(f"❌ 处理 {filename} 失败: {e}")

    # 保存 CSV 报告
    with open(os.path.join(OUTPUT_DIR, "icbhi_binary_report.csv"), "w") as f:
        f.write("filename,prediction,confidence\n")
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
    print(f"✨ 复现完成！报告保存在: {OUTPUT_DIR}")