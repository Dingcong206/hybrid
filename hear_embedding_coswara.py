import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# 核心：引入你提到的官方工具包
# 请确保你的 sys.path 包含 hear 源码路径，或者该文件在当前目录下
try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError:
    print("❌ 错误：找不到 hear.python.data_processing.audio_utils。请确保 HeAR 源码在路径中。")

# ================= 路径与参数 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_v1")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_segmented.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 严格标签映射
STRICT_LABEL_MAP = {
    'healthy': 0,
    'positive_mild': 1,
    'positive_moderate': 1,
    'positive_asymp': 1
}


def main():
    print(f"🚀 正在加载 HeAR 官方模型及音频工具类...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析 Coswara 元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：扫描音频文件 ---
    audio_tasks = []
    print(f"🔍 正在扫描 Coswara 音频文件...")
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                u_id = os.path.basename(root).strip()
                # 确保是长字符串 ID
                if len(u_id) < 15: continue

                status = label_lookup.get(u_id)
                if status in STRICT_LABEL_MAP:
                    audio_tasks.append({
                        "u_id": u_id,
                        "f_name": f,
                        "path": os.path.join(root, f),
                        "label": STRICT_LABEL_MAP[status],
                        "status": status
                    })

    print(f"📊 扫描完成：共发现 {len(audio_tasks)} 个符合条件的有效音频。")

    # --- 第三步：利用官方 audio_utils 提取特征 ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 特征提取")

    with torch.no_grad():
        for task in audio_tasks:
            u_id = task["u_id"]
            wav_path = task["path"]
            f_name = task["f_name"]

            try:
                # 1. 加载原始音频
                waveform, sr = torchaudio.load(wav_path)

                # 2. 调用 HeAR 官方预处理 (保留检测器截取逻辑)
                # 使用你提供的正确入口：audio_utils.preprocess_audio
                # 它会自动处理检测、重采样、截断或补齐至 32000 点
                spec = audio_utils.preprocess_audio(waveform).to(DEVICE)

                # 3. 提取进入 ViT 之前的 Patch Embeddings
                # 直接通过 model.embeddings 获得 [1, 97, 1024]
                # 这里会返回一个元组，我们根据你之前的报错信息处理它
                output = model.embeddings(spec)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 转换为 Numpy 保存 (97, 1024)
                feature_np = x.squeeze(0).cpu().numpy()

                # 4. 保存文件与元数据
                save_name = f"{u_id}_{f_name.replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feature_np)

                meta_data.append({
                    "user_id": u_id,
                    "original_wav": f_name,  # 对齐训练脚本列名
                    "feature_path": save_path,
                    "label": task["label"],
                    "covid_status": task["status"]
                })

            except Exception as e:
                # pbar.write(f"⚠️ 跳过 {u_id}/{f_name}: {e}")
                pass

            pbar.update(1)

    pbar.close()

    # 保存最终 CSV
    if meta_data:
        df_out = pd.DataFrame(meta_data)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\n✅ 成功生成 {len(df_out)} 个特征文件。")
        print(f"📊 阳性样本比例: {df_out['label'].mean():.2%}")
    else:
        print("❌ 提取结果为空，请检查 audio_utils 是否正常工作。")


if __name__ == "__main__":
    main()