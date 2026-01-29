import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# 必须引入官方工具包，这是检测器的核心
try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError:
    print("❌ 错误：未找到 hear 官方工具包。请确保 sys.path 包含 hear 源码路径。")

# ================= 1. 环境配置 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_patches_expert")
OUT_CSV = os.path.join(BASE_DIR, "coswara_metadata_expert.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LABEL_MAP = {'healthy': 0, 'positive_mild': 1, 'positive_moderate': 1, 'positive_asymp': 1}


def main():
    print(f"🚀 初始化 HeAR 专家模式 (声学检测器 + ViT 拦截)...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：扫描文件 ---
    audio_tasks = []
    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue
        for f in files:
            if f.endswith(('.wav', '.webm')):
                parts = root.split(os.sep)
                u_id = next((p for p in parts if len(p) > 20), None)
                if u_id and u_id in label_lookup:
                    status = label_lookup[u_id]
                    if status in LABEL_MAP:
                        audio_tasks.append(
                            {"u_id": u_id, "f_name": f, "path": os.path.join(root, f), "label": LABEL_MAP[status]})

    print(f"📊 扫描完成：共 {len(audio_tasks)} 条录音。开始专家级提取...")

    # --- 第三步：核心提取逻辑 ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks))

    # 统计跳过原因
    stats = {"detector_skip": 0, "file_error": 0, "success": 0}

    with torch.no_grad():
        for task in audio_tasks:
            try:
                # 1. 安全加载（处理读取权限或损坏问题）
                if not os.access(task["path"], os.R_OK):
                    stats["file_error"] += 1
                    continue

                waveform, sr = torchaudio.load(task["path"])

                # 2. 论文核心：调用官方声学检测器
                # preprocess_audio 内部会进行：
                # 检测声学事件 -> 截取最有意义的段 -> 重采样至 16k -> 对齐至 32000 点
                spec = audio_utils.preprocess_audio(waveform)

                if spec is None:
                    # 这就是为什么之前会“提取失败”：检测器认为这段音频是废片（纯静音或纯噪音）
                    stats["detector_skip"] += 1
                    continue

                # 3. 提取 Patch Embeddings (拦截进入 ViT 之前)
                audio_input = spec.to(DEVICE)
                output = model.embeddings(audio_input)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 4. 保存 [97, 1024]
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
                stats["success"] += 1

            except Exception as e:
                stats["file_error"] += 1
                continue
            finally:
                pbar.update(1)

    pbar.close()

    # --- 第四步：保存统计 ---
    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"\n✨ 提取总结:")
        print(f"✅ 成功生成: {stats['success']} 个特征")
        print(f"🚫 检测器过滤 (无意义音频): {stats['detector_skip']} 个")
        print(f"⚠️ 文件读取错误/权限问题: {stats['file_error']} 个")
    else:
        print("❌ 任务结束，未生成任何特征。请检查检测器逻辑。")


if __name__ == "__main__":
    main()