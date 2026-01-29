import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# 核心：引入官方工具包
try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError:
    print("❌ 错误：找不到 hear.python.data_processing.audio_utils。请检查路径。")

# ================= 1. 路径与参数 =================
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
    print(f"🚀 正在加载 HeAR 模型 (Device: {DEVICE})...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析 Coswara 元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：扫描音频文件 ---
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

    print(f"📊 扫描完成：匹配到 {len(audio_tasks)} 个有效任务。")

    # --- 第三步：特征提取 (核心修复 Rank 2 问题) ---
    meta_data = []
    error_log = []
    pbar = tqdm(total=len(audio_tasks), desc="HeAR 特征提取")

    with torch.no_grad():
        for task in audio_tasks:
            u_id, f_name, wav_path = task["u_id"], task["f_name"], task["path"]

            try:
                # 1. 加载音频并强制修复 Rank (必须为 Rank 2: [1, samples])
                waveform, sr = torchaudio.load(wav_path)

                # 修正 Rank 问题
                if waveform.ndim == 1:
                    # [samples] -> [1, samples]
                    waveform = waveform.unsqueeze(0)
                elif waveform.shape[0] > 1:
                    # 多声道 -> 单声道 [1, samples]
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # 此时 waveform 形状必定是 [1, samples]，满足 Rank 2 要求

                # 2. 调用官方检测器逻辑
                # 它会自动处理内部重采样和健康声学段截取
                spec = audio_utils.preprocess_audio(waveform).to(DEVICE)

                if spec is None:
                    # 检测器未发现有效声学事件，静默跳过
                    pbar.update(1)
                    continue

                # 3. 提取 Patch Embeddings (进入 ViT 之前)
                output = model.embeddings(spec)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 4. 转换并保存
                feature_np = x.squeeze(0).cpu().numpy()  # 结果为 (97, 1024)
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
                # 仅记录前几个错误用于调试
                if len(error_log) < 5:
                    error_log.append(f"文件 {u_id}/{f_name} 出错: {e}")

            finally:
                pbar.update(1)

    pbar.close()

    # 打印调试阶段捕获的错误
    if error_log:
        print("\n⚠️ 运行期间捕获的错误样例:")
        for err in error_log:
            print(f"  - {err}")

    # --- 第四步：保存元数据 CSV ---
    if meta_data:
        df_out = pd.DataFrame(meta_data)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\n✨ 任务完成！")
        print(f"✅ 成功生成特征文件: {len(df_out)}")
        print(f"📊 过滤/跳过样本数: {len(audio_tasks) - len(df_out)}")
        print(f"📄 索引文件已更新: {OUT_CSV}")
    else:
        print("\n❌ 提取失败：生成文件数为 0。请检查上方错误样例。")


if __name__ == "__main__":
    main()