import numpy as np
import os

# 1. 随意取一个文件
data_dir = "/data/dingcong/hybrid/segmented_patches_v1"
files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

if not files:
    print("目录里没找到文件！")
else:
    sample_file = files[0]
    file_path = os.path.join(data_dir, sample_file)

    # 2. 加载数据
    data = np.load(file_path)

    print(f"--- 文件检查报告 ---")
    print(f"文件名: {sample_file}")
    print(f"特征维度 (Shape): {data.shape}")

    # 3. 维度验证逻辑
    if data.shape == (97, 1024):
        print("✅ 维度正确：这是 97 个 Raw Patch，符合进入 ViT 之前的特征。")
    elif data.shape == (98, 1024):
        print("⚠️ 包含 CLS Token：维度是 98，后续 Mamba 训练时记得用 data[1:, :]。")
    else:
        print(f"❌ 维度异常：拿到的维度是 {data.shape}，请检查提取拦截点。")

    # 4. 数值检查（确保不是全 0 或空值）
    print(f"数值均值: {data.mean():.4f}")
    print(f"数值标准差: {data.std():.4f}")