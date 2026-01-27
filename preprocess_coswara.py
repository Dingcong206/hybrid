import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置区 (针对服务器上的 Coswara) =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
# 数据解压后的根目录
EXTRACTED_DIR = os.path.join(BASE_DIR, "Extracted_data")
# 标签 CSV 文件
LABEL_CSV = os.path.join(BASE_DIR, "combined_data.csv")
# 输出保存 NPY 的目录
SAVE_DIR = os.path.join(BASE_DIR, "coswara_spec_npy")
# 输出新的 metadata.csv
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

SR = 16000
DURATION = 5  # 咳嗽音频通常较短，5秒足够覆盖一次完整的咳嗽动作
N_MELS = 128
TARGET_LEN = 1024
# 动态调整步长以匹配 1024 宽度
HOP_LENGTH = int(SR * DURATION / (TARGET_LEN - 1))


# =====================================================

def extract_spectrogram(y):
    """将音频片段转换为标准的 Log-Mel 频谱"""
    # 提取 Mel 频谱
    spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=HOP_LENGTH
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # 归一化
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)

    # 裁剪/填充到目标宽度 1024
    if spec_db.shape[1] > TARGET_LEN:
        spec_db = spec_db[:, :TARGET_LEN]
    else:
        pad_width = TARGET_LEN - spec_db.shape[1]
        spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
    return spec_db


def run_preprocessing():
    # 1. 加载官方标签
    if not os.path.exists(LABEL_CSV):
        print(f"❌ 错误: 找不到标签文件 {LABEL_CSV}")
        return
    df_labels = pd.read_csv(LABEL_CSV)

    # 定义正负样本标签逻辑 (根据 Coswara 状态)
    # 确诊或无症状阳性定为 1，健康定为 0
    pos_statuses = ['covid_confirmed', 'positive_asymp', 'positive_mild', 'positive_moderate']
    neg_statuses = ['healthy']

    metadata = []

    # 2. 扫描那 43 个日期文件夹
    date_folders = [f for f in os.listdir(EXTRACTED_DIR) if os.path.isdir(os.path.join(EXTRACTED_DIR, f))]
    print(f"检测到 {len(date_folders)} 个日期文件夹，开始处理咳嗽音频...")

    for date_dir in tqdm(date_folders):
        date_path = os.path.join(EXTRACTED_DIR, date_dir)
        # 遍历每个受试者 ID
        for user_id in os.listdir(date_path):
            user_path = os.path.join(date_path, user_id)
            if not os.path.isdir(user_path): continue

            # 查找重度咳嗽音频
            audio_file = os.path.join(user_path, "cough-heavy.wav")
            if not os.path.exists(audio_file):
                continue

            # 3. 匹配标签
            user_info = df_labels[df_labels['id'] == user_id]
            if user_info.empty: continue

            status = user_info.iloc[0]['covid_status']
            if status in pos_statuses:
                final_label = 1
            elif status in neg_statuses:
                final_label = 0
            else:
                continue  # 排除‘康复中’或‘待定’等模糊标签

            # 4. 加载并处理音频
            try:
                # 统一加载 DURATION 秒
                y, _ = librosa.load(audio_file, sr=SR, duration=DURATION)
                if len(y) < SR * 0.5: continue  # 过滤掉短于0.5秒的无效录音

                spec = extract_spectrogram(y)

                # 5. 保存 .npy
                npy_name = f"{user_id}_heavy.npy"
                np.save(os.path.join(SAVE_DIR, npy_name), spec.astype(np.float32))

                metadata.append({
                    'wav_name': npy_name.replace('.npy', '.wav'),  # 适配你 train.py 的逻辑
                    'label': final_label,
                    'user_id': user_id
                })
            except Exception as e:
                # 针对损坏的音频文件跳过
                continue

    # 6. 保存新的 metadata.csv
    new_df = pd.DataFrame(metadata)
    new_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ 预处理完成！")
    print(f"生成的样本总数: {len(new_df)}")
    print(f"类别分布:\n{new_df['label'].value_counts()}")


if __name__ == "__main__":
    run_preprocessing()