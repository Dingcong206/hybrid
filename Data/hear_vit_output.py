
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib
import torch
from transformers import AutoModel

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--hear-pytorch/snapshots/"
    "f791cd42437c3e268c8ac84707e3508900f65f1a"
)

HEAR_REPO = "/data/dingcong/hybrid/hear"

def main():
    # 让 python 能 import 你本地 hear repo 的 preprocess
    os.environ["PYTHONPATH"] = HEAR_REPO + ":" + os.environ.get("PYTHONPATH", "")

    # 导入 hear 的 preprocess_audio（官方示例就是这个）
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    preprocess_audio = audio_utils.preprocess_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("MODEL_DIR:", MODEL_DIR)

    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    print("✅ model loaded")

    # 假 2 秒音频： (B, 32000)
    wav = torch.randn(2, 32000, dtype=torch.float32)

    # 前处理（wave -> spec/features）
    feats = preprocess_audio(wav)  # 通常是 (B, F, T) 或类似
    print("preprocess_audio output shape:", tuple(feats.shape))

    with torch.no_grad():
        out = model(feats.to(device), return_dict=True, output_hidden_states=True)

    # 打印可用字段
    if hasattr(out, "keys"):
        print("output keys:", list(out.keys()))

    # 常见候选：ViT token 序列 / 池化 embedding
    for name in ["pooler_output", "embeddings", "last_hidden_state"]:
        if hasattr(out, name) and getattr(out, name) is not None:
            t = getattr(out, name)
            print(f"{name} shape:", tuple(t.shape))

    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        print("num hidden_states:", len(out.hidden_states))
        print("hidden_states[-1] shape:", tuple(out.hidden_states[-1].shape))

if __name__ == "__main__":
    main()

