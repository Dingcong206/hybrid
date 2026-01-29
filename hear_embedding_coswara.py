import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

# ================= 1. 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 根据你的 cut 命令输出结果适配的标签映射
STRICT_LABEL_MAP = {
    'healthy': 0,
    'positive_mild': 1,
    'positive_moderate': 1,
    'positive_asymp': 1
}


def main():
    print(f"🚀 正在加载 HeAR 官方模型及健康声学检测器...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析 Coswara 元数据 ---
    if not os.path.exists(COSWARA_CSV):
        print(f"❌ 未找到标签文件: {COSWARA_CSV}")
        return
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：深度扫描音频文件 ---
    audio_tasks = []
    print(f"🔍 正在扫描全量解压后的音频文件...")

    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root:
            continue

        for f in files:
            if f.endswith(('.wav', '.webm')):
                # 核心修正：获取当前文件夹名
                user_id = os.path.basename(root).strip()

                # 如果文件夹名是日期（长度通常为8），则 ID 可能在上一层或这一层不对
                # Coswara 的 ID 通常是长字符串（28位）
                if len(user_id) < 15:
                    continue

                status = label_lookup.get(user_id)
                if status in STRICT_LABEL_MAP:
                    full_path = os.path.join(root, f)
                    label = STRICT_LABEL_MAP[status]
                    audio_tasks.append((user_id, f, full_path, label, status))

    # --- 调试检查：防止输出为 0 ---
    if len(audio_tasks) == 0:
        print("❌ 匹配失败！未找到任何匹配标签的音频。")
        print(f"检查：CSV 中前 3 个 ID 分别是: {list(label_lookup.keys())[:3]}")
        print(f"检查：最后扫描到的文件夹名是: {os.path.basename(root)}")
        return

    print(f"📊 匹配成功：共发现 {len(audio_tasks)} 个有效音频样本。")

    # --- 第三步：特征提取 ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 特征提取进度")

    with torch.no_grad():
        for user_id, file_name, wav_path, label, status in audio_tasks:
            try:
                waveform, sr = torchaudio.load(wav_path)
                # HeAR 预处理：包含健康声学段检测
                processed_audio = model.preprocess_audio(waveform, sr).to(DEVICE)
                # 获取 Patch Embedding
                patch_embeddings = model.embeddings(processed_audio)

                feature_np = patch_embeddings.squeeze(0).cpu().numpy()
                safe_file_name = file_name.replace('.', '_')
                save_path = os.path.join(SAVE_DIR, f"{user_id}_{safe_file_name}.npy")
                np.save(save_path, feature_np)

                meta_data.append({
                    "user_id": user_id,
                    "original_file": file_name,
                    "feature_path": save_path,
                    "label": label,
                    "covid_status": status
                })
            except Exception:
                pass
            pbar.update(1)

    pbar.close()

    # --- 第四步：保存 ---
    df_out = pd.DataFrame(meta_data)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n✅ 任务完成！提取样本数: {len(df_out)}")
    print(f"📄 索引文件已生成: {OUT_CSV}")


if __name__ == "__main__":
    main()