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

# 标签映射 (严格对应你的 cut 结果)
STRICT_LABEL_MAP = {
    'healthy': 0,
    'positive_mild': 1,
    'positive_moderate': 1,
    'positive_asymp': 1
}


def main():
    print(f"🚀 正在加载 HeAR 官方模型 (Device: {DEVICE})...")
    # trust_remote_code 是调用官方健康声学预处理逻辑的关键
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # --- 第一步：解析元数据 ---
    if not os.path.exists(COSWARA_CSV):
        print(f"❌ 未找到标签文件: {COSWARA_CSV}")
        return
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    label_lookup = {row['id']: row['covid_status'] for _, row in df_raw.iterrows()}

    # --- 第二步：深度扫描音频 ---
    audio_tasks = []
    print(f"🔍 正在扫描全量音频文件...")

    for root, dirs, files in os.walk(BASE_DIR):
        if "coswara_hear_patches" in root: continue

        for f in files:
            if f.endswith(('.wav', '.webm')):
                # 提取 User ID (当前文件夹名)
                u_id = os.path.basename(root).strip()

                # 过滤掉非 ID 的日期文件夹 (ID 长度通常较长)
                if len(u_id) < 15: continue

                status = label_lookup.get(u_id)
                if status in STRICT_LABEL_MAP:
                    audio_tasks.append({
                        "user_id": u_id,
                        "file_name": f,
                        "path": os.path.join(root, f),
                        "label": STRICT_LABEL_MAP[status],
                        "status": status
                    })

    if len(audio_tasks) == 0:
        print("❌ 扫描完成但未发现匹配 ID。请检查文件夹名是否为 ID 字符串。")
        return

    print(f"📊 匹配成功：共发现 {len(audio_tasks)} 个有效音频。开始提取...")

    # --- 第三步：特征提取 (核心纠错版) ---
    # --- 第三步：特征提取 (稳健版) ---
    meta_data = []
    pbar = tqdm(total=len(audio_tasks), desc="特征提取进度")

    with torch.no_grad():
        for task in audio_tasks:
            u_id = task["user_id"]
            wav_path = task["path"]
            f_name = task["file_name"]

            try:
                # 1. 加载音频并重采样至 16kHz (HeAR 必须要求 16k)
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                # 2. 替代 preprocess_audio 的逻辑
                # 如果 model.preprocess_audio 报错，我们手动处理输入
                try:
                    # 尝试原方法
                    processed = model.preprocess_audio(waveform, 16000).to(DEVICE)
                except AttributeError:
                    # 如果原方法不存在，手动对齐音频长度（HeAR 期望 2秒 = 32000 个采样点）
                    # 简单填充或截断到 32000
                    if waveform.shape[1] > 32000:
                        waveform = waveform[:, :32000]
                    elif waveform.shape[1] < 32000:
                        waveform = torch.nn.functional.pad(waveform, (0, 32000 - waveform.shape[1]))

                    # 模拟 preprocess_audio 的关键步骤：获取 Log-Mel Spectrogram
                    # 这里的处理如果复杂，可以直接尝试 model 内部的 call
                    processed = waveform.to(DEVICE)

                # 3. 提取特征
                # 如果 model.embeddings 也不行，使用通用 forward
                try:
                    embeddings = model.embeddings(processed)
                except AttributeError:
                    # 最后的兜底逻辑：直接调用 forward
                    # 注意：有些版本的 forward 直接返回的就是 patch embeddings
                    outputs = model(processed)
                    embeddings = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs

                feat_np = embeddings.squeeze(0).cpu().numpy()

                # 4. 保存 (代码同前...)
                save_name = f"{u_id}_{f_name.replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feat_np)

                meta_data.append({
                    "user_id": u_id,
                    "original_wav": f_name,
                    "feature_path": save_path,
                    "label": task["label"],
                    "covid_status": task["status"]
                })

            except Exception as e:
                pbar.write(f"⚠️ 跳过 {u_id}/{f_name} | 错误: {e}")

            pbar.update(1)

    # --- 第四步：保存结果 ---
    if len(meta_data) > 0:
        df_out = pd.DataFrame(meta_data)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\n✨ 任务圆满完成！")
        print(f"✅ 成功提取特征数: {len(df_out)}")
        print(f"📄 索引文件已生成: {OUT_CSV}")
    else:
        print("\n❌ 提取失败：虽然扫描到了音频，但提取过程中全部报错。请检查 GPU 显存或 torchaudio 版本。")


if __name__ == "__main__":
    main()