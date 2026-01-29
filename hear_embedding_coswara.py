import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import traceback

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
    print(f"🚀 正在加载 HeAR 模型并启用官方健康声学检测器...")
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

    print(f"📊 匹配成功：{len(audio_tasks)} 个音频。开始利用 HeAR 检测器提取 Patch...")

    # --- 第三步：特征提取 (核心修正) ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="提取进度")

    with torch.no_grad():
        for u_id, f_name, wav_path, label, status in audio_tasks:
            try:
                # 1. 加载音频
                waveform, sr = torchaudio.load(wav_path)

                # 2. **调用 HeAR 官方检测器逻辑** # preprocess_audio 会自动处理重采样、静音切除，并截取探测器认为有意义的声学段
                # 如果 model.preprocess_audio 依然因为环境问题报错，这块需要根据 model 实际属性微调
                try:
                    # 这里的 processed 已经是探测器截取后的 [1, N_samples] 音频
                    processed_audio = model.preprocess_audio(waveform, sr).to(DEVICE)
                except AttributeError:
                    # 如果顶层没有 preprocess_audio，则尝试调用其内部 detector 模块
                    # 这里的逻辑根据 google/hear-pytorch 的最新远程代码适配
                    processed_audio = model.audio_detector(waveform, sr).to(DEVICE)

                # 3. **提取进入 ViT 之前的 Patch Embeddings**
                # 我们避开 model.embeddings()，直接调用其底层的 ViT embedding 模块
                # 在 ViT 架构中，这一层负责将 Log-Mel 频谱切块并线性映射
                if hasattr(model, 'vit'):
                    patch_outputs = model.vit.embeddings(processed_audio)
                else:
                    # 最后的安全方案：如果无法定位 vit 子模块，则手动通过 embeddings 但处理元组
                    output = model.embeddings(processed_audio)
                    patch_outputs = output[0] if isinstance(output, (tuple, list)) else output

                # 此时形状应为 [1, 97, 1024]
                feat_np = patch_outputs.squeeze(0).cpu().numpy()

                # 4. 保存
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
        print(f"✅ 提取成功！样本数: {len(meta_data)}")
    else:
        print("❌ 提取失败。")


if __name__ == "__main__":
    main()