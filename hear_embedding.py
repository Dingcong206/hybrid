import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

# ================= 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid"
WAV_DIR = os.path.join(BASE_DIR, "audio_and_txt_files")
SAVE_DIR = os.path.join(BASE_DIR, "segmented_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 1. 核心提取主程序 =================
def main():
    print(f"正在加载 HeAR 官方模型组件...")
    # 直接加载模型，确保 trust_remote_code=True 以加载官方预处理逻辑
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析标注文件，确定总数 ---
    # 这是根据你的要求：先根据文件给出总共要提取的呼吸段数量
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
    all_tasks = []
    total_segments_count = 0

    print(f"🔍 正在解析标注文件...")
    for wav_name in wav_files:
        txt_name = wav_name.replace(".wav", ".txt")
        txt_path = os.path.join(WAV_DIR, txt_name)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                total_segments_count += len(lines)
                all_tasks.append((wav_name, txt_path, lines))

    print(f"📊 统计完成：共发现 {len(all_tasks)} 个音频文件，包含 {total_segments_count} 个呼吸段")

    # --- 第二步：使用 HeAR 官方逻辑进行特征提取 ---
    meta_data = []
    pbar = tqdm(total=total_segments_count, desc="HeAR 特征提取")

    with torch.no_grad():
        for wav_name, txt_path, lines in all_tasks:
            wav_path = os.path.join(WAV_DIR, wav_name)
            waveform, sr = torchaudio.load(wav_path)

            for i, line in enumerate(lines):
                try:
                    parts = line.split('\t')
                    if len(parts) < 4: continue

                    start_t, end_t = float(parts[0]), float(parts[1])
                    crackle, wheeze = int(parts[2]), int(parts[3])

                    # 1. 裁剪原始音频段
                    start_sample = int(start_t * sr)
                    end_sample = int(end_t * sr)
                    audio_chunk = waveform[:, start_sample:end_sample]

                    # 2. 调用 HeAR 官方预处理
                    # 这里会执行 HeAR 的健康声学检测重采样、2秒对齐等所有官方步骤
                    # 注意：model.preprocess_audio 是 HeAR 封装的官方入口
                    spec = model.preprocess_audio(audio_chunk, sr).to(DEVICE)

                    # 3. 拦截：进入 ViT 之前的 Embedding 层
                    # 这一步拿到的就是经过 HeAR 官方 Patchify 和 Positional Encoding 后的特征
                    x = model.embeddings(spec)

                    # 转换为 Numpy 保存 (97, 1024)
                    feature_np = x.squeeze(0).cpu().numpy()

                    # 4. 保存文件与元数据
                    seg_id = f"{wav_name.replace('.wav', '')}_seg_{i}"
                    save_path = os.path.join(SAVE_DIR, f"{seg_id}.npy")
                    np.save(save_path, feature_np)

                    label = 1 if (wheeze == 1 or crackle == 1) else 0
                    meta_data.append({
                        "original_wav": wav_name,
                        "segment_id": seg_id,
                        "feature_path": save_path,
                        "label": label
                    })
                    pbar.update(1)

                except Exception as e:
                    pbar.write(f"⚠️ 出错 {wav_name} 第 {i} 段: {e}")
                    pbar.update(1)

    pbar.close()

    # 保存新的 CSV
    df = pd.DataFrame(meta_data)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n--- 任务完成 ---")
    print(f"✅ 成功生成 {len(df)} 个特征文件。")
    print(f"📂 特征保存在: {SAVE_DIR}")
    print(f"📊 异常样本比例: {df['label'].mean():.2%}")


if __name__ == "__main__":
    main()