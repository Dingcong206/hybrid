import sys
import os
import torch

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mymodels.model import TimeFrequencyEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeFrequencyEncoder(
    token_dim=768,
    freq_patches=12,
    time_patches=79,
    time_depth=2,
    freq_depth=2,
    num_heads=8,
    dropout=0.1
).to(device)

x = torch.randn(2, 948, 768).to(device)

feature = model(x)

print("input shape:", x.shape)
print("feature shape:", feature.shape)