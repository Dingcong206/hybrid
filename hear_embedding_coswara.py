import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel
import sys

# 必须确保你能导入 hear 源码里的工具包
# 如果 hear 文件夹在 /data/dingcong/hybrid/hear，请确保它在路径里
sys.path.append("/data/dingcong/hybrid")

try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError:
    print("❌ 无法导入 hear.python.data_processing.audio_utils，请确认 hear 源码路径")

# ================= 1. 配置 =================
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
    print(f"🚀 正在启动 HeAR 专家模式 (带声学检测器)...")

    # 加载模型
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # 读取元数据
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # 扫描任务
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

    print(f"📊 匹配成功：{len(audio_tasks)} 个音频。开始执行检测与提取...")

    meta_data = []
    # 增加计数器用于诊断
    stats = {"success": 0, "skip_no_event": 0, "file_broken": 0}

    pbar = tqdm(total=len(audio_tasks))

    with torch.no_grad():
        for task in audio_tasks:
            try:
                # 1. 显式尝试读取音频，处理“权限/损坏”问题
                if not os.path.exists(task["path"]): continue

                # 针对损坏文件做一次 try，避免中断整个循环
                try:
                    waveform, sr = torchaudio.load(task["path"])
                except Exception:
                    stats["file_broken"] += 1
                    continue

                # 2. 调用论文核心：声学检测预处理器
                # 这个函数会做：AED探测、切片、重采样、对齐
                # 输入要求 rank 2: [1, samples]
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # preprocess_audio 是论文中提到的关键步骤
                spec = audio_utils.preprocess_audio(waveform)

                if spec is None:
                    # 如果检测器没找到有效声学事件（比如全是噪音），跳过
                    stats["skip_no_event"] += 1
                    continue

                # 3. 提取进入 ViT 之前的 Patch Embeddings
                audio_input = spec.to(DEVICE)  # 此时 spec 已经是符合论文的 [1, 32000]
                output = model.embeddings(audio_input)
                x = output[0] if isinstance(output, (tuple, list)) else output

                # 4. 保存 (97, 1024)
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
                # 打印异常详情以备排查
                # print(f"Error on {task['f_name']}: {e}")
                stats["file_broken"] += 1

            finally:
                pbar.update(1)

    pbar.close()

    # 保存结果
    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"\n✨ 提取总结:")
        print(f"✅ 成功特征数: {stats['success']}")
        print(f"🚫 检测器过滤(无有效音): {stats['skip_no_event']}")
        print(f"❌ 文件损坏/读取失败: {stats['file_broken']}")
    else:
        print("❌ 错误：最终未生成任何特征。请检查音频路径。")


if __name__ == "__main__":
    main()