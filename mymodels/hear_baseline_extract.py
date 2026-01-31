import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# --- 1. 严格路径配置 (直接从你的 image_d7d3c6.png 复制) ---
# 请确保这个快照 ID 是最新的
SNAPSHOT_PATH = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

# 根据 ls 结果，子文件夹名称必须完全一致
FRONTEND_PATH = os.path.join(SNAPSHOT_PATH, "spectrogram_frontend")
ENCODER_PATH = SNAPSHOT_PATH  # 根目录存放基础编码器
DETECTOR_PATH = os.path.join(SNAPSHOT_PATH, "event_detector_large")

# 数据路径
DATA_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUT_DIR = "/data/dingcong/hybrid/features/icbhi_final_binary"
os.makedirs(OUT_DIR, exist_ok=True)

# 路径自检
for p in [FRONTEND_PATH, ENCODER_PATH, DETECTOR_PATH]:
    if not os.path.exists(os.path.join(p, "saved_model.pb")):
        print(f"❌ 警告：路径 {p} 下找不到 saved_model.pb，请检查文件名！")

# --- 2. 加载模型组件 ---
print("🔗 正在加载 HeAR 官方组件...")
frontend = tf.saved_model.load(FRONTEND_PATH).signatures["serving_default"]
encoder = tf.saved_model.load(ENCODER_PATH).signatures["serving_default"]
detector = tf.saved_model.load(DETECTOR_PATH).signatures["serving_default"]


# --- 3. 官方复现：ICBHI 二分类处理逻辑 ---
def predict_icbhi(audio_path):
    # HeAR 官方方法：先检测整段音频，找到最显著的事件点
    audio, _ = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # 全局扫描定位
    spec = list(frontend(audio=waveform).values())[0]
    emb = list(encoder(x=spec).values())[0]
    logits = list(detector(x=emb).values())[0]

    # 找到得分最高的时间点
    scores = tf.reduce_max(logits, axis=-1).numpy()[0]
    best_idx = np.argmax(scores)

    # 官方裁剪：以此点为中心截取 2 秒 (32000点)
    center = int((best_idx / len(scores)) * len(audio))
    start = max(0, center - 16000)
    end = min(len(audio), start + 32000)

    crop = audio[start:end]
    if len(crop) < 32000:
        crop = np.pad(crop, (0, 32000 - len(crop)))

    # 最终二分类：对 2s 精华片段进行判定
    crop_tensor = tf.convert_to_tensor(crop[np.newaxis, :], dtype=tf.float32)
    f_out = list(frontend(audio=crop_tensor).values())[0]
    e_out = list(encoder(x=f_out).values())[0]
    d_out = list(detector(x=e_out).values())[0]

    # 获取最高概率分数 (0-1)
    prob = np.max(tf.nn.sigmoid(d_out).numpy())
    label = 1 if prob > 0.5 else 0

    return label, prob


# --- 4. 批量预测 ---
if __name__ == "__main__":
    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    results = []

    print(f"🚀 开始对 {len(wav_files)} 个 ICBHI 样本进行官方复现推理...")
    with tf.device('/GPU:0'):
        for f in tqdm(wav_files):
            try:
                l, p = predict_icbhi(os.path.join(DATA_DIR, f))
                results.append(f"{f},{l},{p:.4f}")
            except Exception as e:
                print(f"❌ {f} 失败: {e}")

    with open(os.path.join(OUT_DIR, "report.csv"), "w") as f:
        f.write("filename,prediction,score\n")
        f.write("\n".join(results))
    print("✨ 处理完成！")