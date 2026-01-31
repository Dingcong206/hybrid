#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import librosa
from tqdm import tqdm
from transformers import AutoModel

# =========================
# 1) 官方路径与环境配置
# =========================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
COSWARA_CSV = os.path.join(BASE_DIR, "combined_data.csv")

# 文件夹命名完全对齐官方
SAVE_DIR = os.path.join(BASE_DIR, "coswara_hear_official_embeddings")
OUT_CSV = os.path.join(BASE_DIR, "coswara_hear_official_metadata.csv")

# 官方模型 ID
HEAR_MODEL_ID = "google/hear-pytorch"
# 官方健康事件检测器 (MobileNet-V3)
DETECTOR_URL = "https://tfhub.dev/google/hear/event_detector/1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000
WINDOW_SIZE = 32000  # 2秒窗口
THRESHOLD = 0.2  # HeAR 官方建议阈值
COUGH_INDEX = 0  # 官方固定索引: 0=Cough, 1=Breathing

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 2) 导入 HeAR 官方包中的预处理逻辑
# =========================
def import_hear_preprocess():
    """
    依赖： hear 源码在 /data/dingcong/hybrid/hear/...
    通过把 /data/dingcong/hybrid 加到 sys.path 来 import
    """
    sys.path.insert(0, "/data/dingcong/hybrid")
    import importlib
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    return audio_utils.preprocess_audio


# =========================
# 3) 核心逻辑：完全复刻官方 Event-Triggered 流程
# =========================

def extract_official_embeddings(audio_path, detector, hear_model):
    """
    1:1 复刻 HeAR 官方 Colab Demo 逻辑
    """
    # 1. 加载音频并重采样至 16kHz (官方强制要求)
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    # 2. 调用官方 Health Detector 扫描整段音频
    # 输出 probabilities: [Num_Seconds, 5]
    outputs = detector(y)
    probs = outputs['probabilities'].numpy()

    # 3. 筛选官方 Index 0 (Cough) 超过阈值的时刻
    valid_secs = np.where(probs[:, COUGH_INDEX] > THRESHOLD)[0]

    # Fallback: 如果整段没检测到咳嗽，取概率最大的一秒 (官方稳健性策略)
    if len(valid_secs) == 0:
        valid_secs = [np.argmax(probs[:, COUGH_INDEX])]

    segment_embeddings = []

    for sec in valid_secs:
        # 以检测到的秒数为起点，截取 2s 片段
        start = int(sec * TARGET_SR)
        end = start + WINDOW_SIZE

        segment = y[start:end]
        if len(segment) < WINDOW_SIZE:
            segment = np.pad(segment, (0, WINDOW_SIZE - len(segment)))

        # 4. 调用 HeAR 官方预处理 (audio_utils)
        seg_tensor = torch.from_numpy(segment).float().unsqueeze(0)
        spec = preprocess_audio(seg_tensor)  # 转为特定尺寸的 Spectrogram

        # 5. 调用 HeAR 官方 Transformer 提取特征
        with torch.no_grad():
            out = hear_model(spec.to(DEVICE), return_dict=True, output_hidden_states=True)
            # 官方特征位置：进入 block 之前的 patch tokens (包含 cls)
            # 我们取 [1, 96, 1024] 的 patch tokens
            tokens = out.hidden_states[0]
            patch_tokens = tokens[:, 1:, :].squeeze(0).cpu().numpy()
            segment_embeddings.append(patch_tokens)

    # 拼接该音频所有被选中的片段特征: [N * 96, 1024]
    return np.concatenate(segment_embeddings, axis=0)


# =========================
# 4) 主执行程序
# =========================

def main():
    print("💎 Loading Official Google Health Detector...")
    detector = hub.load(DETECTOR_URL)

    print("💎 Loading Official HeAR Backbone (ViT)...")
    hear_model = AutoModel.from_pretrained(HEAR_MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

    # 读取任务列表 (复用你之前的 CSV 解析逻辑)
    df_labels = pd.read_csv(COSWARA_CSV)
    # ... (此处省略你已有的文件扫描代码以保持简洁) ...
    # 假设扫描后的任务存放在 audio_tasks 列表中

    results = []
    for task in tqdm(audio_tasks, desc="Official Processing"):
        try:
            # 执行官方全套管线
            final_feat = extract_official_embeddings(task["path"], detector, hear_model)

            save_name = f"{task['user_id']}_{task['fname'].replace('.', '_')}.npy"
            save_path = os.path.join(SAVE_DIR, save_name)
            np.save(save_path, final_feat)

            results.append({
                "user_id": task["user_id"],
                "label": task["label"],
                "feature_path": save_path,
                "segments_found": final_feat.shape[0] // 96,
                "status": "success"
            })
        except Exception as e:
            print(f"❌ Failed: {task['fname']} | {e}")

    # 保存最终索引表
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n✅ 任务完成！")
    print(f"📍 特征保存至: {SAVE_DIR}")
    print(f"📍 索引表保存至: {OUT_CSV}")


if __name__ == "__main__":
    main()