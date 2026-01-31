import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# --- 1. 路径配置 (使用你 GitHub 仓库中的本地模型) ---
REPO_BASE_DIR = "/data/dingcong/hybrid/hear"
DATASET_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
OUTPUT_DIR = "/data/dingcong/hybrid/features/icbhi_official_binary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载官方组件
hear_model = tf.saved_model.load(REPO_BASE_DIR)
hear_func = hear_model.signatures["serving_default"]
frontend_func = tf.saved_model.load(os.path.join(REPO_BASE_DIR, "frontend")).signatures["serving_default"]
detector_func = tf.saved_model.load(os.path.join(REPO_BASE_DIR, "event_detector")).signatures["serving_default"]


def process_official_hear_flow(audio_path):
    # A. 加载原始长音频
    audio, sr = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # B. 全局检测：找出音频中“最异常”的位置
    # 链路：Frontend -> Encoder -> Detector
    f_out = frontend_func(audio=waveform)['spectrogram']  # 根据签名调整key
    e_out = hear_func(x=f_out)['embedding']
    d_out = detector_func(x=e_out)['logits']  # 得到 [1, Time, Classes]

    # C. 官方裁剪逻辑：寻找事件中心
    # 假设异常音标签在 detector 输出的某些维度上，我们取最大激活时刻
    event_scores = tf.reduce_max(d_out, axis=-1).numpy()[0]  # 聚合各异常类的得分
    best_step = np.argmax(event_scores)  # 概率最高的时刻

    # 计算中心点对应的采样点位置 (HeAR 的 time step 通常对应 20ms-50ms)
    # 这里根据 HeAR 官方 stride 换算，假设 best_step 对应的时间点是 t_center
    total_steps = len(event_scores)
    t_center_ratio = best_step / total_steps
    center_sample = int(t_center_ratio * len(audio))

    # 截取以 center_sample 为中心的 2 秒片段 (32000 个采样点)
    start = max(0, center_sample - 16000)
    end = min(len(audio), start + 32000)
    # 如果靠后不够截，往前挪
    if end == len(audio): start = max(0, end - 32000)

    official_crop = audio[start:end]
    if len(official_crop) < 32000:  # 补齐
        official_crop = np.pad(official_crop, (0, 32000 - len(official_crop)))

    # D. 最终判定：对这 2 秒精华片段进行二分类
    crop_tensor = tf.convert_to_tensor(official_crop[np.newaxis, :], dtype=tf.float32)
    final_f = frontend_func(audio=crop_tensor)['spectrogram']
    final_e = hear_func(x=final_f)['embedding']
    final_d = detector_func(x=final_e)['logits']

    # 最终异常得分：该 2s 片段的平均最大概率
    final_score = np.mean(tf.reduce_max(final_d, axis=-1).numpy())
    prediction = 1 if final_score > 0.5 else 0

    return prediction, final_score


# --- 2. 批量处理 ---
def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    results = []
    print(f"🚀 按照 HeAR 官方检测+裁剪逻辑处理 ICBHI...")

    with tf.device('/GPU:0'):
        for filename in tqdm(files):
            try:
                pred, score = process_official_hear_flow(os.path.join(DATASET_DIR, filename))
                results.append(f"{filename},{pred},{score:.4f}")
            except Exception as e:
                print(f"❌ {filename} 失败: {e}")

    with open(os.path.join(OUTPUT_DIR, "official_binary_results.csv"), "w") as f:
        f.write("filename,label,score\n")
        f.write("\n".join(results))


if __name__ == "__main__":
    main()