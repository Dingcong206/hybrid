import matplotlib.pyplot as plt


def verify_patch_visually(npy_path):
    data = np.load(npy_path)  # 加载 (N, 190, 256)
    # 取第一个 2 秒片段的第一个 Patch 序列
    sample_patch = data[0]

    plt.figure(figsize=(10, 4))
    # 转置是为了让时间轴在横轴
    plt.imshow(sample_patch.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Patch Visualization: {os.path.basename(npy_path)}")
    plt.xlabel("Time Steps (Patches)")
    plt.ylabel("Feature Dimensions")
    plt.colorbar()
    plt.show()


# 调用检查
verify_patch_visually("/data/dingcong/hybrid/hear_patches_data/138_1p3_Ll_mc_AKGC417L.npy")