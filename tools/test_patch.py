import numpy as np  # <--- 必须补上这一行！
import matplotlib.pyplot as plt
import os


def verify_patch_visually(npy_path):
    # 加载提取出来的 npy 文件
    data = np.load(npy_path)

    print(f"📊 文件名: {os.path.basename(npy_path)}")
    print(f"📏 Patch 维度: {data.shape}")

    # 预期维度应该是 (N, 190, 256)
    # N 是 2 秒片段的数量
    # 190 是时间步，256 是每个 Patch 的特征维度

    if len(data.shape) == 3:
        # 取第一个 2 秒片段的可视化
        sample = data[0]
        plt.figure(figsize=(10, 4))
        # .T 是转置，让横轴代表时间，纵轴代表特征
        plt.imshow(sample.T, aspect='auto', origin='lower', cmap='magma')
        plt.title(f"HeAR Patch Visualization: {os.path.basename(npy_path)}")
        plt.xlabel("Time Steps (Patches)")
        plt.ylabel("Feature Dimensions")
        plt.colorbar(label="Intensity")
        plt.show()
    else:
        print(f"❌ 维度不对！当前维度为 {data.shape}，这看起来像是 Embedding 而不是 Patch。")


# 运行验证
verify_patch_visually("/data/dingcong/hybrid/hear_patches_data/138_1p3_Ll_mc_AKGC417L.npy")