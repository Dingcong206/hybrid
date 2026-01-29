import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

# ================= 1. 路径与参数 (已根据你的环境适配) =================
# 原始数据总目录
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
# 官方标签 CSV 路径
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")

# 输出目录：特征文件 (.npy)
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
# 输出目录：提取后的索引 CSV
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 严格标签过滤映射
# 只保留确定的健康 (0) 和 阳性 (1)，剔除 recovered 或 unidentified 等模糊项
STRICT_LABEL_MAP = {
    'healthy': 0,
    'positive_mild': 1,
    'positive_moderate': 1,
    'positive_asymp': 1
}


# ================= 2. 核心提取主程序 =================
def main():
    print(f"🚀 正在加载 HeAR 官方模型及健康声学检测器...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析 Coswara 元数据 ---
    if not os.path.exists(COSWARA_CSV):
        print(f"❌ 未找到标签文件: {COSWARA_CSV}")
        return
    df_raw = pd.read_csv(COSWARA_CSV)
    # 清洗 ID 字符串防止空格干扰
    df_raw['id'] = df_raw['id'].astype(str).str.strip()

    # 建立一个快速查找字典
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：深度扫描音频文件 ---
    audio_tasks = []
    print(f"🔍 正在扫描全量解压后的音频文件 (BASE_DIR: {BASE_DIR})...")

    # 使用 os.walk 递归寻找所有 .wav/.webm
    for root, dirs, files in os.walk(BASE_DIR):
        # 排除存放提取结果的目录，防止循环扫描
        if "coswara_hear_patches" in root:
            continue

        for f in files:
            if f.endswith(('.wav', '.webm')):
                # 关键：由于解压结构是 20200413/UserID/file.wav
                # os.path.basename(root) 提取的就是 User ID
                user_id = os.path.basename(root).strip()

                # 检查该 ID 是否在 CSV 中且标签是否符合“严格模式”
                status = label_lookup.get(user_id)
                if status in STRICT_LABEL_MAP:
                    full_path = os.path.join(root, f)
                    label = STRICT_LABEL_MAP[status]
                    audio_tasks.append((user_id, f, full_path, label, status))

    print(f"📊 扫描完成：共发现 {len(audio_tasks)} 个符合条件的有效音频。")

    # --- 第三步：使用 HeAR 检测器并获取 Patch ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 特征提取进度")

    with torch.no_grad():
        for user_id, file_name, wav_path, label, status in audio_tasks:
            try:
                # 1. 加载原始音频
                waveform, sr = torchaudio.load(wav_path)

                # 2. 调用 HeAR 官方预处理 (包含健康声学段检测与截取)
                # 输出 shape 适合 ViT 输入
                processed_audio = model.preprocess_audio(waveform, sr).to(DEVICE)

                # 3. 拦截：获取 Patch Embedding ( ViT 入口前)
                # 输出 shape: [1, 97, 1024]
                patch_embeddings = model.embeddings(processed_audio)

                # 4. 保存结果为二进制 .npy
                feature_np = patch_embeddings.squeeze(0).cpu().numpy()
                # 文件名使用 ID+原始文件名，防止冲突
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

            except Exception as e:
                # pbar.write(f"⚠️ 跳过 {file_name} (ID: {user_id}): {e}")
                pass
            pbar.update(1)

    pbar.close()

    # --- 第四步：保存生成的训练索引 ---
    df_out = pd.DataFrame(meta_data)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n--- 任务全部完成 ---")
    print(f"✅ 成功从 22GB 数据中提取出 {len(df_out)} 个高质量样本。")
    print(f"📂 特征存储位置: {SAVE_DIR}")
    print(f"📄 训练索引文件: {OUT_CSV}")


if __name__ == "__main__":
    main()