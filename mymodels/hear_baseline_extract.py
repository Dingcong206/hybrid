import os
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import librosa

# 1. 加载模型（会自动使用你下载好的缓存）
print("🔗 正在从缓存加载 HeAR 模型...")
model = from_pretrained_keras("google/hear")
# 获取推理签名
infer = model.signatures["serving_default"]


# 2. 定义处理函数
def extract_features(audio_path):
    # HeAR 通常要求 16000Hz 采样率
    audio, _ = librosa.load(audio_path, sr=16000)
    # 增加 batch 维度并转换为 tensor
    audio_tensor = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)

    # 进行推理
    output = infer(audio_tensor)

    # HeAR 会输出多种特征，通常我们取 'embedding' (1024维)
    # 具体 key 名取决于模型，刚才检查脚本输出过
    embedding = output['embedding'].numpy()
    return embedding


# 3. 测试运行
test_file = "audio_and_txt_files"  # 确保这个路径有音频文件
if os.path.exists(test_file):
    feat = extract_features(test_file)
    print(f"✅ 特征提取成功！形状为: {feat.shape}")
    # 保存特征
    np.save("test_feature.npy", feat)
else:
    print("❌ 未找到测试音频，请检查路径。")