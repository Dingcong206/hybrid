import torch
import torch.nn.functional as F


def patch_10_200_48_to_tokens(patch, seq_len=96):
    """
    将 HeAR spectrogram patch (10, 200, 48)
    转换为 SSA 可用的 token 序列 (seq_len, 48)

    Args:
        patch: torch.Tensor, shape (10, 200, 48)
        seq_len: int, 默认 96

    Returns:
        torch.Tensor, shape (seq_len, 48)
    """
    assert patch.shape == (10, 200, 48), f"Expected (10,200,48), got {patch.shape}"

    # 1️⃣ 频率维压缩：200 -> 10
    x = patch.permute(0, 2, 1)           # (10, 48, 200)
    x = F.adaptive_avg_pool1d(x, 10)     # (10, 48, 10)
    x = x.permute(0, 2, 1)               # (10, 10, 48)

    # 2️⃣ 展平成 token：10×10 = 100
    x = x.reshape(100, 48)               # (100, 48)

    # 3️⃣ 压缩到 seq_len（默认 96）
    if x.shape[0] != seq_len:
        x = x.T.unsqueeze(0)             # (1, 48, 100)
        x = F.adaptive_avg_pool1d(x, seq_len)
        x = x.squeeze(0).T               # (96, 48)

    return x
