import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置区 (Linux 路径) =================
BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")
SAVE_DIR = os.path.join(BASE_DIR, "spec_npy_v2")
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

SR = 16000  # 采样率
DURATION = 8  # 统一填充到8秒（确保能容纳绝大多数呼吸周期）
N_MELS = 128  # 频率轴
TARGET_LEN = 1024  # 时间轴宽度
HOP_LENGTH = int(SR * DURATION / (TARGET_LEN - 1))  # 固定步长，保证尺寸一致


# =====================================================

def extract_spectrogram(y):
    """将音频片段转换为标准的 Log-Mel 频谱"""
    # 提取 Mel 频谱
    spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=HOP_LENGTH
    )
    # 转为 DB 刻度
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # 归一化
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)
    # 裁剪/填充到目标宽度
    if spec_db.shape[1] > TARGET_LEN:
        spec_db = spec_db[:, :TARGET_LEN]
    else:
        pad_width = TARGET_LEN - spec_db.shape[1]
        spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
    return spec_db


def run_preprocessing():
    metadata = []
    files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    print(f"检测到 {len(files)} 个原始音频文件，开始切分周期...")

    for f in tqdm(files):
        file_id = f.replace('.wav', '')
        wav_path = os.path.join(WAV_DIR, f)
        txt_path = os.path.join(WAV_DIR, file_id + '.txt')

        if not os.path.exists(txt_path):
            continue

        # 1. 加载完整音频
        y_full, _ = librosa.load(wav_path, sr=SR)

        # 2. 读取标注文件 (Start, End, Crackles, Wheezes)
        try:
            # ICBHI 标注格式通常是：开始时间 结束时间 是否有啰音 是否有哮鸣音
            annotations = pd.read_csv(txt_path, sep='\t', header=None)
        except Exception as e:
            print(f"跳过文件 {f}，读取标注出错: {e}")
            continue

        # 3. 按行切分呼吸周期
        for i, row in annotations.iterrows():
            start_t, end_t = row[0], row[1]
            c_label, w_label = int(row[2]), int(row[3])

            # 确定标签：只要有啰音或哮鸣音，就标记为异常(1)
            final_label = 1 if (c_label == 1 or w_label == 1) else 0

            # 裁剪音频片段
            start_s = int(start_t * SR)
            end_s = int(end_t * SR)
            y_seg = y_full[start_s:end_s]

            if len(y_seg) < 100:  # 过滤极短的无效片段
                continue

            # 4. 转换为频谱
            spec = extract_spectrogram(y_seg)

            # 5. 保存 .npy
            npy_name = f"{file_id}_seg_{i}.npy"
            np.save(os.path.join(SAVE_DIR, npy_name), spec.astype(np.float32))

            # 6. 存入元数据
            metadata.append({
                'wav_name': npy_name.replace('.npy', '.wav'),  # 保持与train.py匹配逻辑
                'label': final_label,
                'original_file': f
            })

    # 保存新的 metadata.csv
    new_df = pd.DataFrame(metadata)
    new_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ 预处理完成！")
    print(f"生成的样本总数: {len(new_df)}")
    print(f"类别分布:\n{new_df['label'].value_counts()}")


if __name__ == "__main__":
    run_preprocessing()