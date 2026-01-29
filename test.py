import numpy as np
import os

# 随机读取一个提取好的文件
feature_path = "/data/dingcong/hybrid/segmented_patches_v1/101_1b1_Al_sc_Meditron_seg_0.npy"
data = np.load(feature_path)

print(f"特征形状: {data.shape}")
if data.shape[0] == 97:
    print("✅ 确认：这是纯粹的 97 个时间补丁序列。")
elif data.shape[0] == 98:
    print("⚠️ 提示：包含 CLS Token，建议后续训练只取 data[1:, :]。")