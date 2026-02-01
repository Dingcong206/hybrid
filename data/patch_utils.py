import torch
import torch.nn.functional as F


def patch_10_200_48_to_tokens(patch, seq_len=96):
    """
    patch: Tensor of shape (T, 200, 48), T 可变
    return: (seq_len, 48)
    """

    assert patch.ndim == 3, f"Expected 3D tensor, got {patch.shape}"
    T, F, C = patch.shape
    assert F == 200 and C == 48, f"Expected (*,200,48), got {patch.shape}"

    # -------- Step 1: 频率维压缩 200 -> 10 --------
    # (T,200,48) -> (T,48,200)
    x = patch.permute(0, 2, 1)

    # (T,48,200) -> (T,48,10)
    x = F.adaptive_avg_pool1d(x, 10)

    # (T,48,10) -> (T,10,48)
    x = x.permute(0, 2, 1)

    # -------- Step 2: 展平成 token --------
    # (T,10,48) -> (T*10,48)
    x = x.reshape(T * 10, 48)

    # -------- Step 3: 时间维统一到 seq_len --------
    # (T*10,48) -> (seq_len,48)
    x = x.transpose(0, 1).unsqueeze(0)     # (1,48,T*10)
    x = F.adaptive_avg_pool1d(x, seq_len)  # (1,48,seq_len)
    x = x.squeeze(0).transpose(0, 1)       # (seq_len,48)

    return x
