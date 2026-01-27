import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置区 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
EXTRACTED_DIR = os.path.join(BASE_DIR, "Extracted_data")
LABEL_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_multi_modal_npy")
OUT_CSV = os.path.join(BASE_DIR, "metadata_multi.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

# 目标模态列表
MODALITIES = ['cough-heavy', 'cough-shallow', 'breathing-deep', 'vowel-a', 'counting-normal']

SR = 16000
DURATION = 5  # 统一 5 秒
N_MELS = 128
TARGET_LEN = 1024
HOP_LENGTH = int(SR * DURATION / (TARGET_LEN - 1))


# ==========================================

def extract_spectrogram(y):
    """音频转 Log-Mel 频谱"""
    spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=HOP_LENGTH
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)

    if spec_db.shape[1] > TARGET_LEN:
        spec_db = spec_db[:, :TARGET_LEN]
    else:
        pad_width = TARGET_LEN - spec_db.shape[1]
        spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
    return spec_db


def run_preprocessing():
    df_labels = pd.read_csv(LABEL_CSV)
    pos_statuses = ['covid_confirmed', 'positive_asymp', 'positive_mild', 'positive_moderate']
    neg_statuses = ['healthy']

    metadata = []
    date_folders = [f for f in os.listdir(EXTRACTED_DIR) if os.path.isdir(os.path.join(EXTRACTED_DIR, f))]

    print(f"开始多模态处理，目标模态: {MODALITIES}")

    for date_dir in tqdm(date_folders):
        date_path = os.path.join(EXTRACTED_DIR, date_dir)
        for user_id in os.listdir(date_path):
            user_path = os.path.join(date_path, user_id)
            if not os.path.isdir(user_path): continue

            # 匹配标签
            user_info = df_labels[df_labels['id'] == user_id]
            if user_info.empty: continue
            status = user_info.iloc[0]['covid_status']

            if status in pos_statuses:
                label = 1
            elif status in neg_statuses:
                label = 0
            else:
                continue

            # 遍历并处理每种声音模态
            available_modes = []
            for mode in MODALITIES:
                audio_file = os.path.join(user_path, f"{mode}.wav")
                # Coswara 有些是 webm 格式，兼容一下
                if not os.path.exists(audio_file):
                    audio_file = audio_file.replace('.wav', '.webm')

                if os.path.exists(audio_file):
                    try:
                        y, _ = librosa.load(audio_file, sr=SR, duration=DURATION)
                        if len(y) < SR * 0.5: continue

                        spec = extract_spectrogram(y)
                        save_name = f"{user_id}_{mode}.npy"
                        np.save(os.path.join(SAVE_DIR, save_name), spec.astype(np.float32))
                        available_modes.append(mode)
                    except:
                        continue

            # 如果该受试者至少有一个有效音频，记录下来
            if available_modes:
                metadata.append({
                    'user_id': user_id,
                    'label': label,
                    'modes': ",".join(available_modes)  # 记录该用户拥有的所有模态
                })

    new_df = pd.DataFrame(metadata)
    new_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ 预处理完成！总计受试者: {len(new_df)}")
    print(f"模态分布统计:\n{new_df['modes'].str.split(',').explode().value_counts()}")


if __name__ == "__main__":
    run_preprocessing()