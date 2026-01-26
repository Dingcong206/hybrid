import os
import pandas as pd

# ================= 配置路径 =================
# 原始标注文件 (.txt) 所在的文件夹
TXT_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
# 刚才生成的 .npy 所在的文件夹（用于核对文件名）
NPY_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\spec_npy_v2"
# 输出 CSV 的保存路径
SAVE_CSV = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\metadata.csv"


# ===========================================

def generate_metadata():
    data = []
    # 1. 找到所有生成的 .npy 文件
    npy_files = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]

    print(f"检测到 {len(npy_files)} 个预处理后的样本，正在提取标签...")

    for npy_name in npy_files:
        # 获取对应的原始文件名（去掉 .npy）
        base_name = npy_name.replace('.npy', '')
        txt_path = os.path.join(TXT_DIR, base_name + '.txt')

        if os.path.exists(txt_path):
            # 2. 读取标注文件
            # ICBHI 标注格式: start, end, crackles, wheezes
            df_ann = pd.read_csv(txt_path, sep='\t', header=None)

            # 3. 判定逻辑：只要出现过 Crackles 或 Wheezes，即为异常 (1)
            # 你也可以根据需要改为四分类：0-Normal, 1-Crackle, 2-Wheeze, 3-Both
            has_crackles = df_ann[2].sum() > 0
            has_wheezes = df_ann[3].sum() > 0

            label = 1 if (has_crackles or has_wheezes) else 0

            data.append({
                'wav_name': base_name + '.wav',  # 对应 Dataset 里的逻辑
                'label': label
            })
        else:
            print(f"⚠️ 找不到标注文件: {txt_path}")

    # 4. 保存为 CSV
    df_final = pd.DataFrame(data)
    df_final.to_csv(SAVE_CSV, index=False)
    print(f"✅ 成功！CSV 已保存至: {SAVE_CSV}")
    print(f"样本分布统计：\n{df_final['label'].value_counts()}")


if __name__ == "__main__":
    generate_metadata()