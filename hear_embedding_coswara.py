import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F
import traceback

# 核心：引入官方工具包
try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError:
    print("❌ 错误：找不到 hear.python.data_processing.audio_utils。请检查 PYTHONPATH。")

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
    'healthy': 0,
    'positive_mild': 1,
    'positive_moderate': 1,
    'positive_asymp': 1
}


def main():
    print(f"🚀 正在加载 HeAR 模型 (Device: {DEVICE})...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：扫描音频 ---
    audio_tasks = []
    print(f"🔍 正在扫描 Coswara 原始音频...")
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                u_id = os.path.basename(root).strip()
                if len(u_id) < 15: continue
                status = label_lookup.get(u_id)
                if status in STRICT_LABEL_MAP:
                    audio_tasks.append({
                        "u_id": u_id,
                        "f_name": f,
                        "path": os.path.join(root, f),
                        "label": STRICT_LABEL_MAP[status]
                    })

    print(f"📊 匹配完成：共 {len(audio_tasks)} 个音频。开始特征提取...")

    # --- 第三步：特征提取 (含维度修复与错误追踪) ---
    meta_data = []
    error_count = 0
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 特征提取")

    with torch.no_grad():
        for task in audio_tasks:
            u_id, f_name, wav_path = task["u_id"], task["f_name"], task["path"]

            try:
                # 1. 加载并预处理维度
                waveform, sr = torchaudio.load(wav_path)

                # 重要修复：audio_utils 通常期望 1D Tensor [samples]
                # 而 torchaudio 默认返回 2D [channels, samples]
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                else:
                    waveform = waveform.squeeze(0)

                # 2. 调用官方检测器逻辑
                # 它会自动处理重采样和有效声学段截取
                spec = audio_utils.preprocess_audio(waveform).to(DEVICE)

                # 如果检测器没找到声音，spec 可能是 None 或触发 AttributeError
                if spec is None:
                    continue

                # 3. 提取 Patch Embeddings (进入 ViT 之前)
                output = model.embeddings(spec)
                # 处理 (embeddings, metadata) 元组
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 4. 转换并保存
                feature_np = x.squeeze(0).cpu().numpy()
                save_name = f"{u_id}_{f_name.replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feature_np)

                meta_data.append({
                    "user_id": u_id,
                    "original_wav": f_name,
                    "feature_path": save_path,
                    "label": task["label"]
                })

            except Exception as e:
                error_count += 1
                # 只打印前 3 个错误原因，防止刷屏
                if error_count <= 3:
                    pbar.write(f"⚠️ 调试信息 | 文件: {u_id}/{f_name} | 错误: {e}")
                continue

            finally:
                pbar.update(1)

    pbar.close()

    # --- 第四步：保存结果 ---
    if meta_data:
        df_out = pd.DataFrame(meta_data)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\n✨ 任务完成！")
        print(f"✅ 成功生成特征文件: {len(df_out)} (跳过/被检测器过滤: {len(audio_tasks) - len(df_out)})")
        print(f"📄 索引文件: {OUT_CSV}")
    else:
        print("\n❌ 提取失败：请检查上方打印的错误信息。")


if __name__ == "__main__":
    main()