import numpy as np
import os

# 替换为你的特征保存目录
feat_dir = "/data/dingcong/hybrid/hear_features_official"
files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]

if files:
    test_file = os.path.join(feat_dir, files[0])
    data = np.load(test_file)

    print(f"📄 检查文件: {files[0]}")
    print(f"📏 特征维度 (Shape): {data.shape}")
    print(f"🔢 数值范围: Max={data.max():.4f}, Min={data.min():.4f}, Mean={data.mean():.4f}")
else:
    print("❌ 文件夹里还没有文件，再等等进度条。")