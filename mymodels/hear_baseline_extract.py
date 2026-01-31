import os
import numpy as np
import tensorflow as tf
import librosa
import json
from tqdm import tqdm

# ==========================================================
# 1. 严格路径配置 (请确保与你的 ls 结果完全一致)
# ==========================================================
# 基础快照路径
BASE_SNAPSHOT = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

# 子组件文件夹
FRONTEND_DIR = os.path.join(BASE_SNAPSHOT, "spectrogram_frontend")
ENCODER_DIR = BASE_SNAPSHOT  # 根目录包含主编码器
DETECTOR_DIR = os.path.join(BASE_SNAPSHOT, "event_detector_large")

# 数据输入与结果输出
DATASET_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUTPUT_DIR = "/data/dingcong/hybrid/features/icbhi_final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. 加载 HeAR 官方模型组件
# ==========================================================
print("🔗 正在从本地路径加载 HeAR 官方组件...")


def load_component(path, name):
    if not os.path.exists(os.path.join(path, "saved_model.pb")):
        raise FileNotFoundError(f"❌ 在 {path} 找不到模型文件，请检查！")
    print(f"✅ 成功定位 {name}")
    return tf.saved_model.load(path).signatures["serving_default"]


try:
    frontend_fn = load_component(FRONTEND_DIR, "前端(Frontend)")
    encoder_fn = load_component(ENCODER_DIR, "编码器(Encoder)")
    detector_fn = load_component(DETECTOR_DIR, "检测器(Detector)")
    print("🚀 所有官方组件加载完毕！")
except Exception as e:
    print(f"💥 加载失败: {e}")
    exit()


# ==========================================================
# 3. 核心推理逻辑：检测器辅助的 2秒 裁剪与预测
# ==========================================================
def run_hear_icbhi_inference(audio_path):
    # A. 采样率标准化 (16kHz)
    audio, _ = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # B. 第一遍：全局扫描定位（官方 Event Detection）
    # 链路：Audio -> Spectrogram -> Embedding -> Event Logits
    spec = list(frontend_fn(audio=waveform).values())[0]
    emb = list(encoder_fn(x=spec).values())[0]
    logits = list(detector_fn(x=emb).values())[0]  # [1, Time, Classes]

    # 找出全段中异常概率最高的时间点
    # 聚合所有异常类别在时间轴上的得分
    time_scores = tf.reduce_max(logits, axis=-1).numpy()[0]
    best_step_idx = np.argmax(time_scores)

    # C. 官方裁剪逻辑：以最显著时刻为中心截取 2 秒 (32000 个采样点)
    center_sample = int((best_step_idx / len(time_scores)) * len(audio))
    start_s = max(0, center_sample - 16000)
    end_s = min(len(audio), start_s + 32000)

    # 保证长度严格等于 2s
    crop = audio[start_s:end_s]
    if len(crop) < 32000:
        crop = np.pad(crop, (0, 32000 - len(crop)))

    # D. 第二遍：对截取的精华 2s 进行最终判定
    crop_tensor = tf.convert_to_tensor(crop[np.newaxis, :], dtype=tf.float32)
    final_spec = list(frontend_fn(audio=crop_tensor).values())[0]
    final_emb = list(encoder_fn(x=final_spec).values())[0]
    final_logits = list(detector_fn(x=final_emb).values())[0]

    # E. 结果映射：二分类判定
    # 使用 sigmoid 将输出映射到 0-1
    probs = tf.nn.sigmoid(final_logits).numpy()
    max_prob = np.max(probs)  # 取该片段中最明显的异常信号强度

    # 1 代表 Abnormal (异常), 0 代表 Normal (正常)
    prediction = 1 if max_prob > 0.5 else 0

    return prediction, float(max_prob)


# ==========================================================
# 4. 批量执行与报告生成
# ==========================================================
def main():
    wav_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    print(f"📂 发现 {len(wav_files)} 个 ICBHI 音频文件，开始复现推理...")

    final_results = []
    # 使用你的 RTX 4090 显卡加速
    with tf.device('/GPU:0'):
        for filename in tqdm(wav_files):
            try:
                audio_path = os.path.join(DATASET_DIR, filename)
                pred, score = run_hear_icbhi_inference(audio_path)

                final_results.append({
                    "file": filename,
                    "prediction": pred,
                    "abnormal_confidence": score
                })
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")

    # 保存为 CSV 格式 (方便查阅)
    csv_path = os.path.join(OUTPUT_DIR, "icbhi_binary_predictions.csv")
    with open(csv_path, "w") as f:
        f.write("filename,prediction(0:Normal 1:Abnormal),confidence\n")
        for item in final_results:
            f.write(f"{item['file']},{item['prediction']},{item['abnormal_confidence']:.4f}\n")

    # 保存为 JSON 格式 (保留原始数据)
    with open(os.path.join(OUTPUT_DIR, "icbhi_binary_predictions.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n✨ 复现成功！结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()