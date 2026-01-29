import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# ================= 1. 配置还原 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LABEL_MAP = {'healthy': 0, 'positive_mild': 1, 'positive_moderate': 1, 'positive_asymp': 1}


def main():
    print(f"🚀 正在还原 HeAR 论文预处理流程 (ViT 前馈拦截)...")
    # trust_remote_code 是必须的，因为 HeAR 的 Spectrogram 转换逻辑写在远程脚本里
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：扫描全量音频 (5.7万样本) ---
    audio_tasks = []
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                u_id = os.path.basename(root).strip()
                if len(u_id) < 15: continue
                status = label_lookup.get(u_id)
                if status in LABEL_MAP:
                    audio_tasks.append(
                        {"u_id": u_id, "f_name": f, "path": os.path.join(root, f), "label": LABEL_MAP[status]})

    # --- 第三步：核心提取 ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="论文级特征提取")

    with torch.no_grad():
        for task in audio_tasks:
            try:
                # 1. 信号对齐论文：16kHz 单声道
                waveform, sr = torchaudio.load(task["path"])
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                # 2. 窗口对齐：强制 2.0s (论文核心参数)
                # 论文提到 313M 的 2s 片段。对于长录音，取中心 2s 是保留呼吸/咳嗽核心的最稳妥办法
                if waveform.shape[1] > 32000:
                    start = (waveform.shape[1] - 32000) // 2
                    waveform = waveform[:, start:start + 32000]
                else:
                    waveform = F.pad(waveform, (0, 32000 - waveform.shape[1]))

                # 3. 维度对齐：确保 Rank 2 [1, 32000]
                audio_input = waveform.to(DEVICE)

                # 4. 拦截：进入 ViT Block 之前的 Embedding
                # 此函数包含：Waveform -> Mel-Spec -> Linear Projection -> + Positional Embedding
                output = model.embeddings(audio_input)

                # 论文中 embeddings 返回通常是 (tensor, metadata)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 5. 保存 [97, 1024]
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
    pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
    print(f"\n✨ 还原完成！共处理 {len(meta_data)} 个样本。")


if __name__ == "__main__":
    main()