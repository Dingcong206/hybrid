import os

feat_dir = "/data/dingcong/hybrid/hear_features_official"
label_dir = "/data/dingcong/hybrid/audio_and_txt_files" # 假设标签和音频在同个目录

feat_files = set([f.replace('.npy', '') for f in os.listdir(feat_dir) if f.endswith('.npy')])
label_files = set([f.replace('.txt', '') for f in os.listdir(label_dir) if f.endswith('.txt')])

# 检查是否有特征但没标签，或者有标签没特征的文件
missing_labels = feat_files - label_files
missing_feats = label_files - feat_files

print(f"✅ 特征文件总数: {len(feat_files)}")
print(f"✅ 标签文件总数: {len(label_files)}")
if missing_labels: print(f"⚠️ 缺失标签的文件: {missing_labels}")
if missing_feats: print(f"⚠️ 缺失特征的文件: {missing_feats}")
import pandas as pd

# 随机读取一个标签文件查看格式
sample_label_path = os.path.join(label_dir, list(label_files)[0] + ".txt")
df = pd.read_csv(sample_label_path, sep='\t', header=None)
df.columns = ['start', 'end', 'crackles', 'wheezes']

print(f"📄 标签样例 ({os.path.basename(sample_label_path)}):")
print(df.head())

# 检查是否有非法值（除了0和1以外的数字）
is_valid = df['crackles'].isin([0, 1]).all() and df['wheezes'].isin([0, 1]).all()
print(f"✔️ 标签值合法性检查: {'通过' if is_valid else '失败'}")