import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# ================= 1. 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

STRICT_LABEL_MAP = {
    'healthy': 0, 'positive_mild': 1, 'positive_moderate': 1, 'positive_asymp': 1
}


def main():
    print(f"🚀 正在加载 HeAR 模型...")
    # 强制 trust_remote_code
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：深度扫描音频 ---
    audio_tasks = []
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                u_id = os.path.basename(root).strip()
                if len(u_id) < 15: continue
                status = label_lookup.get(u_id)
                if status in STRICT_LABEL_MAP:
                    audio_tasks.append((u_id, f, os.path.join(root, f), STRICT_LABEL_MAP[status], status))

    print(f"📊 匹配成功：{len(audio_tasks)} 个音频。开始提取...")

    # --- 第三步：特征提取 (核心修正：解决 Unpack 错误) ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="提取进度")

    with torch.no_grad():
        for u_id, f_name, wav_path, label, status in audio_tasks:
            try:
                # 1. 音频标准化 (16kHz)
                waveform, sr = torchaudio.load(wav_path)
                if waveform.shape[0] > 1:  # 转单声道
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                # 2. 长度截断/填充 (2秒 = 32000采样点)
                if waveform.shape[1] > 32000:
                    waveform = waveform[:, :32000]
                else:
                    waveform = F.pad(waveform, (0, 32000 - waveform.shape[1]))

                # 3. 核心：通过 model.embeddings 提取，并处理元组返回
                audio_input = waveform.to(DEVICE)

                # 【关键修正点】
                output = model.embeddings(audio_input)

                # 如果返回的是元组 (tuple)，提取第一个元素
                if isinstance(output, (tuple, list)):
                    # 报错 expected 4, got 2 说明返回了 (embeddings, something_else)
                    embeddings = output[0]
                else:
                    embeddings = output

                # 4. 确认形状并保存
                feat_np = embeddings.squeeze(0).cpu().numpy()  # 应为 (97, 1024)

                save_name = f"{u_id}_{f_name.replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feat_np)

                meta_data.append({
                    "user_id": u_id,
                    "original_wav": f_name,
                    "feature_path": save_path,
                    "label": label,
                    "covid_status": status
                })

            except Exception as e:
                pbar.write(f"⚠️ 跳过 {u_id}/{f_name} | 错误: {e}")

            pbar.update(1)

    pbar.close()

    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"✅ 成功提取 {len(meta_data)} 个特征文件！")
    else:
        print("❌ 最终提取数为 0，请检查报错详情。")


if __name__ == "__main__":
    main()