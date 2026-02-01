import torch
import torch.nn.functional as F


def patch_10_200_48_to_tokens(patch, seq_len=96):
    """
    patch: Tensor (T, 200, 48), T 可变
    return: (seq_len, 48)
    """
    assert patch.ndim == 3, f"Expected 3D tensor, got {patch.shape}"
    T, Freq, C = patch.shape
    assert Freq == 200 and C == 48, f"Expected (*,200,48), got {patch.shape}"

    # 1) 频率压缩：200 -> 10
    x = patch.permute(0, 2, 1)          # (T,48,200)
    x = F.adaptive_avg_pool1d(x, 10)    # (T,48,10)
    x = x.permute(0, 2, 1)              # (T,10,48)

    # 2) 展平 token：L = T*10
    x = x.reshape(T * 10, 48)           # (T*10,48)

    # 3) 压到 seq_len
    x = x.transpose(0, 1).unsqueeze(0)  # (1,48,T*10)
    x = F.adaptive_avg_pool1d(x, seq_len)
    x = x.squeeze(0).transpose(0, 1)    # (seq_len,48)

    return x
