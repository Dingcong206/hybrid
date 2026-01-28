import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel

# ================= 路径配置 =================
BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")  # 原始音频位置
SAVE_DIR = os.path.join(BASE_DIR, "spec_npy_v2")  # 特征保存位置
OUT_CSV = os.path.join(BASE_DIR, "metadata.csv")  # 记录映射关系

# 确保 HEAR 库能被正确加载 (根据你图片显示的目录结构)
HEAR_PATH = os.path.join(BASE_DIR, "hear")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ================= 参数配置 =================
MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000  # HeAR 要求的采样率
TARGET_LEN = 32000  # HeAR 要求的固定长度 (2秒)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 核心工具函数 =================

def get_most_informative_segment(waveform, target_len=32000):
    C, T = waveform.shape
    if T <= target_len:
        # 确保 pad 后仍然是 [1, target_len]
        return F.pad(waveform, (0, target_len - T))

    energy = waveform.pow(2)
    window_sum = F.avg_pool1d(energy.unsqueeze(0), kernel_size=target_len, stride=1600)
    best_idx = torch.argmax(window_sum).item() * 1600
    start = min(best_idx, T - target_len)

    # 返回 [1, target_len]
    return waveform[:, start: start + target_len]

# ================= 提取主程序 =================

def main():
    print(f"🚀 Loading HeAR model to {DEVICE}...")
    # 1. 加载模型 (trust_remote_code 必须为 True 以加载本地定义的层)
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"❌ 模型加载失败，请检查网络或路径: {e}")
        return

    # 2. 动态导入 HeAR 预处理工具
    try:
        import hear.python.data_processing.audio_utils as audio_utils
        preprocess_audio = audio_utils.preprocess_audio
        print("✅ HEAR 预处理模块加载成功")
    except ImportError:
        print("❌ 找不到 hear 文件夹。请确保 hear 文件夹在 /data/dingcong/hybrid 下")
        return

    # 3. 扫描 WAV 文件
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    meta_data = []

    print(f"📦 Found {len(wav_files)} files. Starting Patch Embedding extraction...")

    with torch.no_grad():
        for filename in tqdm(wav_files):
            wav_path = os.path.join(WAV_DIR, filename)

            try:
                # A. 加载音频
                waveform, sr = torchaudio.load(wav_path)

                # B. 统一重采样到 16kHz
                if sr != TARGET_SR:
                    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

                # C. 智能截取能量最大的 2 秒 (32000 个采样点)
                # 这一步解决了 ValueError: Input audio must have 32000 samples 的报错
                waveform_seg = get_most_informative_segment(waveform, TARGET_LEN)

                # D. 预处理为 Spectrogram
                # preprocess_audio 接收 [Time] 形状
                #spec = preprocess_audio(waveform_seg.squeeze(0)).to(DEVICE)
                spec = preprocess_audio(waveform_seg).to(DEVICE)

                # E. 【核心截取】：提取 Patch Embeddings
                # 这一步会跳过 Transformer Encoder，直接拿到进入模型前的特征块
                patch_embeddings = model.embeddings(spec)

                # F. 转换为 numpy 并保存
                feature_np = patch_embeddings.squeeze(0).cpu().numpy()  # [Seq, Dim]

                save_filename = filename.replace(".wav", ".npy")
                save_path = os.path.join(SAVE_DIR, save_filename)
                np.save(save_path, feature_np)

                # G. 记录到元数据
                # 这里的 label 可以后续根据你的原始 CSV 进行 map
                meta_data.append({
                    "user_id": filename.split('_')[0],
                    "feature_path": save_path,
                    "label": 0  # 默认占位符
                })

            except Exception as e:
                print(f"⚠️ 处理文件 {filename} 时出错: {e}")
                continue
    print(f"DEBUG: Total samples in meta_data: {len(meta_data)}")
    # 4. 保存映射表
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)
    print(f"---")
    print(f"✅ 特征提取完成！")
    print(f"📂 特征保存在: {SAVE_DIR}")
    print(f"📄 元数据保存在: {OUT_CSV}")
    print(f"💡 下一步：请运行你的 train1.py 开始训练 SSA_Model。")


if __name__ == "__main__":
    main()