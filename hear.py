from huggingface_hub import from_pretrained_keras
import os

# 1. 自动从 Hugging Face 下载 HeAR 官方模型
print("正在从 Hugging Face 下载 HeAR 官方权重...")
model = from_pretrained_keras("google/hear")

# 2. 打印出它下载到的真实物理路径
# 这里的 model 其实就是一个加载好的 Keras 模型对象
print("\n✅ 下载成功！")
print(f"模型已加载，它是从 Hugging Face 自动同步的。")