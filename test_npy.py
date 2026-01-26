import os
import numpy as np

# 你的路径
NPY_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\spec_npy_v2"


def check_disk_dimensions(data_dir, num_samples=5):
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not files:
        print("❌ 错误：文件夹里没找到 .npy 文件！")
        return

    print(f"✅ 找到 {len(files)} 个文件。开始抽样验证...")

    for i in range(min(num_samples, len(files))):
        file_path = os.path.join(data_dir, files[i])
        data = np.load(file_path)
        print(f"文件: {files[i]} | 维度: {data.shape}")

        # 深度验证：确保没有空值
        if np.isnan(data).any():
            print(f"⚠️ 警告：文件 {files[i]} 包含 NaN 值！")


check_disk_dimensions(NPY_DIR)