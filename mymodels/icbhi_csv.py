
import os
import pandas as pd

# 1. 设置路径（请确认这是你存放 wav 和 txt 的真实路径）
data_dir = "/data/dingcong/hybrid/audio_and_txt_files"
save_path = "/data/dingcong/hybrid/labels.csv"

# 2. 准备存储列表
all_data = []

# 3. 遍历目录下所有的 txt 文件
txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

for f in txt_files:
    file_path = os.path.join(data_dir, f)
    # ICBHI 的 txt 格式通常是: [start, end, crackle, wheeze]
    # 使用空格或制表符分隔
    try:
        # 读取标注信息
        df_temp = pd.read_csv(file_path, sep='\t', header=None,
                              names=['start', 'end', 'crackle', 'wheeze'])

        # 将文件名（不带后缀）和标签关联起来
        # 这里取该文件所有呼吸循环中，是否有任何一个带有啰音或哮鸣音作为简化标签
        has_crackle = 1 if df_temp['crackle'].sum() > 0 else 0
        has_wheeze = 1 if df_temp['wheeze'].sum() > 0 else 0

        # 对应你生成的 .npy 文件名
        npy_name = f.replace('.txt', '.npy')

        all_data.append({
            'file_name': npy_name,
            'crackle': has_crackle,
            'wheeze': has_wheeze,
            'label': 1 if (has_crackle or has_wheeze) else 0  # 简单分类：异常为1，正常为0
        })
    except Exception as e:
        print(f"跳过文件 {f}: {e}")

# 4. 保存为 CSV
final_df = pd.DataFrame(all_data)
final_df.to_csv(save_path, index=False)

print(f"✅ 成功！已生成 {len(all_data)} 条数据的标注表：{save_path}")
print("   现在你可以去 train1.py 里把 CSV_PATH 指向这个文件了。")
