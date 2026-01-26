import os
import librosa
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
WAV_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"  # 原始音频路径
SAVE_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\spec_npy_v2"  # 新生成的npy存放路径
os.makedirs(SAVE_DIR, exist_ok=True)

SR = 16000  # 采样率：16k是医疗音频处理的标准
DURATION = 8  # 统一长度：8秒（ICBHI音频平均长度）
N_MELS = 128  # 频率分辨率：对应 HeAR 的“频率宽”特点
TARGET_LEN = 1024  # 时间轴长度：对应模型输入的 Width


# ==========================================

def process_single_audio(file_path):
    # 1. 加载音频并统一采样率
    y, sr = librosa.load(file_path, sr=SR)

    # 2. 长度填充或裁剪
    target_samples = SR * DURATION
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    # 3. 提取 Mel 频谱 (关键参数控制)
    # n_fft 和 hop_length 决定了时间轴的分辨率
    # 这里我们让 8秒音频对应 1024 个时间步
    hop_len = int(len(y) / (TARGET_LEN - 1))
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=hop_len)

    # 4. 转化为 Log 刻度（分贝值），符合人类听觉动态范围
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # 5. 归一化 (标准化到 -1 到 1 之间，有利于模型收敛)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)

    # 6. 确保尺寸正好是 (128, 1024)
    return spec_db[:, :TARGET_LEN]


if __name__ == "__main__":
    print("开始生成声学各向异性频谱图...")
    files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    for f in tqdm(files):
        try:
            path = os.path.join(WAV_DIR, f)
            spec = process_single_audio(path)
            save_path = os.path.join(SAVE_DIR, f.replace('.wav', '.npy'))
            np.save(save_path, spec.astype(np.float32))
        except Exception as e:
            print(f"处理文件 {f} 出错: {e}")
    print(f"预处理完成！数据已保存至: {SAVE_DIR}")