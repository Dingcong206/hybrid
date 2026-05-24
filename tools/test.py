import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.full_model import TimeFrequencyLogisticModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeFrequencyLogisticModel(
    token_dim=768,
    freq_patches=12,
    time_patches=79,
    time_depth=2,
    freq_depth=2,
    num_heads=8,
    dropout=0.1,
    num_classes=4
).to(device)

x = torch.randn(2, 948, 768).to(device)

logits = model(x)

print("input shape:", x.shape)
print("logits shape:", logits.shape)