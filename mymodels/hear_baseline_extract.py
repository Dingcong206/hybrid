import os
import numpy as np
import tensorflow as tf
import librosa
import csv
from tqdm import tqdm

# ==========================================================
# 1. 严格路径配置 (基于你提供的 ls -R 输出)
# ==========================================================
# 快照根目录 (Encoder 所在位置)
SNAP_ROOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

# 根据你的输出，Frontend 和 Detector 都在 event_detector 子文件夹下
FRONTEND_PATH = os.path.join(SNAP_ROOT, "event_detector/spectrogram_frontend")
ENCODER_PATH = SNAP_ROOT
# 这里选 large 版本，分类性能更好
DETECTOR_PATH = os.path.join(SNAP_ROOT, "event_detector/event_detector_large")

# 数据与输出路径
DATA_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUT_DIR = "/data/dingcong/hybrid/features/icbhi_final_binary"
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================
# 2. 组件加载函数
# ==========================================================
def load_comp(path, name):
    if not os.path.exists(os.path.join(path, "saved_model.pb")):
        raise FileNotFoundError(f"❌ 路径错误：{path} 没找到 saved_model.pb")
    print(f"✅ 成功加载 {name}")
    return tf.saved_model.load(path).signatures["serving_default"]


print("🔗 正在按层级加载 HeAR 官方组件...")
frontend = load_comp(FRONTEND_PATH, "Spectrogram Frontend")
encoder = load_comp(ENCODER_PATH, "HEAR Encoder")
detector = load_comp(DETECTOR_PATH, "Event Detector (Large)")


# ==========================================================
# 3. 官方复现推理逻辑
# ==========================================================
def predict_icbhi_binary(audio_path):
    # A. 采样率重采样至 16kHz
    audio, _ = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # B. 第一遍扫描：找到音频中最显著的时刻 (定位)
    # 输入 Key 可能是 'audio' 或 'input_1'，此处动态获取
    f_key = list(frontend.structured_input_signature[1].keys())[0]
    spec = list(frontend(**{f_key: waveform}).values())[0]

    e_key = list(encoder.structured_input_signature[1].keys())[0]
    emb = list(encoder(**{e_key: spec}).values())[0]

    d_key = list(detector.structured_input_signature[1].keys())[0]
    logits = list(detector(**{d_key: emb}).values())[0]  # [1, Time, Classes]

    # 找到得分最高的时刻点
    time_scores = tf.reduce_max(logits, axis=-1).numpy()[0]
    best_step = np.argmax(time_scores)

    # C. 官方裁剪：以此点为中心截取 2s (32000点)
    center = int((best_step / len(time_scores)) * len(audio))
    start = max(0, center - 16000)
    end = min(len(audio), start + 32000)
    crop = audio[start:end]
    if len(crop) < 32000:
        crop = np.pad(crop, (0, 32000 - len(crop)))

    # D. 第二遍扫描：对 2s 精华片段进行最终预测
    crop_tensor = tf.convert_to_tensor(crop[np.newaxis, :], dtype=tf.float32)
    f_out = list(frontend(**{f_key: crop_tensor}).values())[0]
    e_out = list(encoder(**{e_key: f_out}).values())[0]
    d_out = list(detector(**{d_key: e_out}).values())[0]

    # E. 判定：1 (Abnormal), 0 (Normal)
    # 计算全类别的 Sigmoid 概率，取最大值
    prob = np.max(tf.nn.sigmoid(d_out).numpy())
    label = 1 if prob > 0.5 else 0

    return label, prob


# ==========================================================
# 4. 批量执行
# ==========================================================
if __name__ == "__main__":
    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"🚀 开始处理 {len(wav_files)} 个 ICBHI 文件...")

    results = []
    with tf.device('/GPU:0'):  # 使用你的 4090
        for f in tqdm(wav_files):
            try:
                l, p = predict_icbhi_binary(os.path.join(DATA_DIR, f))
                results.append([f, l, f"{p:.4f}"])
            except Exception as e:
                print(f"❌ 处理 {f} 失败: {e}")

    # 保存报告
    csv_file = os.path.join(OUT_DIR, "icbhi_prediction_report.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])
        writer.writerows(results)

    print(f"✨ 复现推理完成！结果已存入: {csv_file}")