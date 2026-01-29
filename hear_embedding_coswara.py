import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

# ================= 配置 =================
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
    # 1. 加载模型
    print(f"🚀 加载 HeAR 模型...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # 2. 检查 CSV
    if not os.path.exists(COSWARA_CSV):
        print(f"❌ 错误：找不到 CSV 文件：{COSWARA_CSV}")
        return

    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}
    print(f"📋 CSV 加载完成，共有 {len(label_lookup)} 个 ID 映射。")

    # 3. 扫描文件 (优化路径逻辑)
    audio_tasks = []
    print(f"🔍 扫描音频文件中...")
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith(('.wav', '.webm')):
                # 寻找路径中符合 ID 特征的部分（长字符串）
                parts = root.split(os.sep)
                u_id = None
                for p in parts:
                    if len(p) > 20:  # Coswara ID 通常很长
                        u_id = p.strip()
                        break

                if u_id and u_id in label_lookup:
                    status = label_lookup[u_id]
                    if status in LABEL_MAP:
                        audio_tasks.append({
                            "u_id": u_id,
                            "f_name": f,
                            "path": os.path.join(root, f),
                            "label": LABEL_MAP[status]
                        })

    if not audio_tasks:
        print("❌ 扫描结果为 0！请检查音频文件是否在 BASE_DIR 下，以及文件夹名是否为 ID。")
        # 打印一个示例路径帮自己排查
        return

    print(f"📊 匹配成功：{len(audio_tasks)} 个音频。开始提取...")

    #

    # 4. 提取循环
    meta_data = []
    fail_reasons = {}
    pbar = tqdm(total=len(audio_tasks))

    with torch.no_grad():
        for task in audio_tasks:
            try:
                waveform, sr = torchaudio.load(task["path"])

                # 预处理
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

                # 2秒窗口
                if waveform.shape[1] > 32000:
                    start = (waveform.shape[1] - 32000) // 2
                    waveform = waveform[:, start:start + 32000]
                else:
                    waveform = F.pad(waveform, (0, 32000 - waveform.shape[1]))

                audio_input = waveform.to(DEVICE)

                # 进入 ViT 前的 Embedding
                output = model.embeddings(audio_input)
                x = output[0] if isinstance(output, (tuple, list)) else output

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

            except Exception as e:
                err_type = type(e).__name__
                fail_reasons[err_type] = fail_reasons.get(err_type, 0) + 1

            finally:
                pbar.update(1)

    pbar.close()

    # 5. 结果汇报
    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"✅ 提取大功告成！生成文件：{len(meta_data)}")
        if fail_reasons:
            print(f"⚠️ 失败统计：{fail_reasons}")
    else:
        print("❌ 最终没有生成任何文件。请检查音频读取权限。")
        # 在 main 函数结束前加入
        if meta_data:
            df_out = pd.DataFrame(meta_data)
            print("\n📊 提取统计（按音频类型）:")
            # 假设文件名中包含类型信息，如 cough-heavy
            df_out['audio_type'] = df_out['original_wav'].apply(lambda x: x.split('.')[0])
            print(df_out['audio_type'].value_counts())


if __name__ == "__main__":
    main()