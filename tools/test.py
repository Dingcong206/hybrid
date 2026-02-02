import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# ============================
# 配置检查路径
# ============================
FEAT_DIR = "/data/dingcong/hybrid/hear_16x256_fixed"
SAVE_REPORT_DIR = "./feature_check_report"
os.makedirs(SAVE_REPORT_DIR, exist_ok=True)


def check_features():
    files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith(".npy")])
    if not files:
        print("❌ 错误：未在目录下找到 .npy 文件。")
        return

    print(f"🔍 发现 {len(files)} 个特征文件，正在随机抽检...")

    # 随机抽选 1 个文件进行深度可视化
    sample_file = random.choice(files)
    data = np.load(os.path.join(FEAT_DIR, sample_file))  # (32768, 256)

    # --- 1. 基础维度与数值检查 ---
    print(f"\n--- 文件分析: {sample_file} ---")
    print(f"维度 (Shape):      {data.shape}")
    print(f"均值 (Mean):       {data.mean():.4f}")
    print(f"标准差 (Std):      {data.std():.4f}")
    print(f"最大/最小值:       {data.max():.4f} / {data.min():.4f}")

    has_nan = np.isnan(data).any()
    print(f"是否有 NaN:        {'❌ 是' if has_nan else '✅ 否'}")

    zero_ratio = np.sum(data == 0) / data.size
    print(f"零值占比:          {zero_ratio:.2%}")

    # --- 2. 可视化：特征激活热图 ---
    # 我们只截取前 1024 个 Token（对应前 64 个 Patch）进行精细观察
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    # 取前 1024 个 Token，转置后 shape 为 (256, 1024)
    # y 轴是 256 维特征，x 轴是时间
    display_data = data[:1024, :].T
    plt.imshow(display_data, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='Activation')
    plt.title(f"Feature Map (First 64 Patches) - {sample_file}")
    plt.ylabel("Feature Dim (256)")
    plt.xlabel("Tokens (Time)")

    # --- 3. 可视化：能量曲线 ---
    # 计算每个 Token 的 L2 范数（代表该时刻的信号强度）
    token_energy = np.linalg.norm(data, axis=1)

    plt.subplot(2, 1, 2)
    plt.plot(token_energy[:2048], color='blue', alpha=0.7)
    plt.title("Token Energy Curve (Temporal Continuity Check)")
    plt.ylabel("L2 Norm")
    plt.xlabel("Tokens")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_REPORT_DIR, "feature_detail.png"))
    print(f"✅ 细节热图已保存至: {SAVE_REPORT_DIR}/feature_detail.png")

    # --- 4. 统计分布检查 ---
    plt.figure(figsize=(10, 5))
    plt.hist(data.flatten(), bins=100, color='green', alpha=0.6)
    plt.title("Overall Value Distribution")
    plt.yscale('log')  # 对数坐标查看长尾分布
    plt.xlabel("Value")
    plt.ylabel("Frequency (Log Scale)")
    plt.savefig(os.path.join(SAVE_REPORT_DIR, "value_distribution.png"))
    print(f"✅ 数值分布图已保存至: {SAVE_REPORT_DIR}/value_distribution.png")

    # --- 5. 跨文件一致性快速扫描 ---
    print("\n--- 跨文件一致性检查 ---")
    shapes = []
    for f in files[:20]:  # 扫描前 20 个
        d = np.load(os.path.join(FEAT_DIR, f))
        shapes.append(d.shape)

    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        print(f"✅ 维度一致性检查通过: 所有文件均为 {list(unique_shapes)[0]}")
    else:
        print(f"⚠️ 警告：发现多种维度，请检查提取逻辑！ {unique_shapes}")


if __name__ == "__main__":
    check_features()