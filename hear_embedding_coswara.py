import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# ================= 1. 环境配置 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 标签映射：合并所有阳性状态为 1，健康为 0
LABEL_MAP = {
    'healthy': 0, 'positive_mild': 1, 'positive_moderate': 1, 'positive_asymp': 1
}


def main():
    print(f"🚀 正在初始化 HeAR 论文级特征提取 (Device: {DEVICE})...")
    # 加载模型，此时我们只需要它的 Embedding 字典
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：全量扫描音频 ---
    audio_tasks = []
    print(f"🔍 正在扫描原始音频库...")
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                u_id = os.path.basename(root).strip()
                if len(u_id) < 15: continue
                status = label_lookup.get(u_id)
                if status in LABEL_MAP:
                    audio_tasks.append({
                        "u_id": u_id,
                        "f_name": f,
                        "path": os.path.join(root, f),
                        "label": LABEL_MAP[status]
                    })

    print(f"📊 匹配完成：共 {len(audio_tasks)} 个音频。开始特征降维提取...")

    # --- 第三步：核心逻辑 - 停在 ViT 之前 ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="特征处理进度")

    with torch.no_grad():
        for task in audio_tasks:
            try:
                # 1. 信号预处理
                waveform, sr = torchaudio.load(task["path"])

                # 转单声道
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # 强制重采样 (16kHz 是 HeAR 的基石)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                # 2. 定长处理：2秒对齐 (32,000 点)
                # 无论音频多长，我们取其核心 2 秒，确保进入 VIT 之前的 Patch 形状一致
                if waveform.shape[1] > 32000:
                    # 取中间段，通常比取开头更能抓到有效声学信息
                    start = (waveform.shape[1] - 32000) // 2
                    waveform = waveform[:, start:start + 32000]
                else:
                    # 不足 2 秒则补零
                    waveform = F.pad(waveform, (0, 32000 - waveform.shape[1]))

                audio_input = waveform.to(DEVICE)  # Shape: [1, 32000]

                # 3. 提取 Patch Embeddings
                # 直接调用 model.embeddings，这会执行 Log-Mel 转换和 Linear Projection
                # 此时输出尚未经过 Transformer Blocks 的深层抽象
                output = model.embeddings(audio_input)

                # 处理模型返回的元组 (embeddings, metadata)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 4. 持久化 (Shape 应为 [97, 1024])
                feature_np = x.squeeze(0).cpu().numpy()
                save_name = f"{task['u_id']}_{task['f_name'].replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feature_np)

                meta_data.append({
                    "user_id": task["u_id"],
                    "original_wav": task["f_name"],
                    "feature_path": save_path,
                    "label": task["label"]
                })

            except Exception:
                continue

            finally:
                pbar.update(1)

    pbar.close()

    # --- 第四步：汇总索引 ---
    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"\n✨ 提取大功告成！")
        print(f"✅ 总计处理样本: {len(meta_data)}")
        print(f"📊 特征形状: {feature_np.shape} (每样本 97 个 Patch, 1024 维)")
    else:
        print("\n❌ 提取失败，请检查文件权限或路径。")


if __name__ == "__main__":
    main()