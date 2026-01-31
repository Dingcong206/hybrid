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