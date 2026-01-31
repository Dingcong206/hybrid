import numpy as np

patch_data = np.load("/data/dingcong/hybrid/hear_patches_data/sample.npy")
print(f"📏 Patch 形状: {patch_data.shape}")

if len(patch_data.shape) != 3:
    print("❌ 维度异常：预期应为 (Batch, Time, Dim)")