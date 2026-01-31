import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def check_extracted_features(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到索引文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 索引文件读取成功，共计 {len(df)} 个片段")

    # 1. 检查维度
    print("\n🔍 正在抽样检查维度...")
    sample_size = min(100, len(df))
    sample_df = df.sample(sample_size)

    correct_shape = (96, 1024)
    errors = 0

    for _, row in tqdm(sample_df.iterrows(), total=sample_size):
        try:
            feat = np.load(row['feature_path'])
            if feat.shape != correct_shape:
                print(f"⚠️ 维度异常: {row['feature_path']} -> {feat.shape}")
                errors += 1
        except Exception as e:
            print(f"❌ 读取失败: {row['feature_path']}, Error: {e}")
            errors += 1

    if errors == 0:
        print(f"✅ 维度检查通过 (抽样 {sample_size} 个)")

    # 2. 统计每个用户的片段分布
    print("\n📈 统计数据分布...")
    seg_counts = df.groupby('user_id')['segment_id'].count()
    print(f"  - 每个用户平均片段数: {seg_counts.mean():.2f}")
    print(f"  - 最少片段数: {seg_counts.min()}")
    print(f"  - 最多片段数: {seg_counts.max()}")

    # 3. 统计标签分布 (片段级别)
    label_counts = df['label'].value_counts()
    pos_ratio = (label_counts.get(1, 0) / len(df)) * 100
    print(f"\n🏷️ 片段级别标签分布:")
    print(f"  - Negative (0): {label_counts.get(0, 0)}")
    print(f"  - Positive (1): {label_counts.get(1, 0)} ({pos_ratio:.2f}%)")

    if pos_ratio < 10:
        print("⚠️ 警告：片段级别正样本比例较低，建议在训练时使用 Focal Loss 或过采样。")


if __name__ == "__main__":
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_multi_segments.csv"
    check_extracted_features(CSV_PATH)