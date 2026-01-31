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