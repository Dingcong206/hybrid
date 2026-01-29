import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# ================= 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
WAV_ROOT = BASE_DIR
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 1. 核心提取主程序 =================
def main():
    print(f"正在加载 HeAR 官方模型及健康声学检测器...")
    # 加载模型，trust_remote_code=True 是调用官方预处理逻辑的关键
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析 Coswara 元数据 ---
    if not os.path.exists(COSWARA_CSV):
        print(f"❌ 未找到 {COSWARA_CSV}")
        return
    df_raw = pd.read_csv(COSWARA_CSV)

    # --- 第二步：扫描音频文件 ---
    audio_tasks = []
    # 遍历所有子目录寻找音频（Coswara 结构复杂，必须用 walk）
    for root, dirs, files in os.walk(WAV_ROOT):
        for f in files:
            if f.endswith(('.wav', '.webm')):
                user_id = os.path.basename(root)
                full_path = os.path.join(root, f)
                audio_tasks.append((user_id, f, full_path))

    print(f"📊 扫描完成：共发现 {len(audio_tasks)} 个音频文件")

    # --- 第三步：使用 HeAR 检测器并获取 Patch ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 健康检测与特征提取")

    with torch.no_grad():
        for user_id, file_name, wav_path in audio_tasks:
            try:
                # 1. 匹配标签
                user_info = df_raw[df_raw['id'] == user_id]
                if user_info.empty: continue
                status = user_info.iloc[0]['covid_status']
                label = 1 if 'positive' in status.lower() else 0

                # 2. 加载原始音频
                waveform, sr = torchaudio.load(wav_path)

                # 3. 调用 HeAR 官方预处理 (包含健康声学检测器)
                # 这个方法会自动进行：重采样至 16kHz -> 健康声学段检测 -> 截取/对齐至 2 秒
                # 最终输出适合 ViT 输入的 Spectrogram
                processed_audio = model.preprocess_audio(waveform, sr).to(DEVICE)

                # 4. 拦截：获取进入 ViT 之前的 Patch Embedding
                # 这一步包含了 Patchify (16x16) 和 Positional Encoding
                # 输出 shape 通常为 [batch, num_patches, embedding_dim] -> [1, 97, 1024]
                patch_embeddings = model.embeddings(processed_audio)

                # 5. 保存结果
                feature_np = patch_embeddings.squeeze(0).cpu().numpy()  # (97, 1024)
                seg_id = f"{user_id}_{file_name.replace('.', '_')}"
                save_path = os.path.join(SAVE_DIR, f"{seg_id}.npy")
                np.save(save_path, feature_np)

                meta_data.append({
                    "user_id": user_id,
                    "original_file": file_name,
                    "feature_path": save_path,
                    "label": label,
                    "covid_status": status
                })

            except Exception as e:
                # pbar.write(f"⚠️ 跳过 {file_name}: {e}")
                pass
            pbar.update(1)

    pbar.close()

    # 保存新的元数据索引
    df_out = pd.DataFrame(meta_data)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n--- 任务完成 ---")
    print(f"✅ 成功提取 {len(df_out)} 个样本的 Patch 特征。")
    print(f"📂 存储路径: {SAVE_DIR}")


if __name__ == "__main__":
    main()