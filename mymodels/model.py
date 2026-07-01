import torch
import torch.nn as nn


# ============================================================
# 尝试导入 Mamba
# 如果没有安装 mamba_ssm，则使用双向 GRU 临时代替
# ============================================================
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


# ============================================================
# Feed Forward Network
# ============================================================
class FeedForward(nn.Module):
    """
    Transformer/Mamba block 中使用的前馈网络。

    输入:
        x: [B, L, D]

    输出:
        x: [B, L, D]
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.1
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Time-Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
    """
    沿时间轴进行建模的 Mamba 模块。

    输入:
        x: [B * Fp, Tp, D]

    其中:
        B  : batch size
        Fp : frequency patch 数量
        Tp : time patch 数量
        D  : token embedding dimension

    对于每一个频率 patch，模型沿时间方向进行扫描。
    """

    def __init__(
        self,
        dim=768,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        # Mamba 前的归一化
        self.norm1 = nn.LayerNorm(dim)

        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_mamba = True

        else:
            # 当环境中没有安装 mamba_ssm 时，
            # 使用双向 GRU 作为占位模块，保证代码能够运行。
            self.mamba = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.use_mamba = False

        self.dropout = nn.Dropout(dropout)

        # FFN 前的归一化
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(
            dim=dim,
            dropout=dropout
        )

    def forward(self, x):
        """
        参数:
            x: [B * Fp, Tp, D]

        返回:
            x: [B * Fp, Tp, D]
        """

        # ----------------------------------------------------
        # Mamba + Residual
        # ----------------------------------------------------
        residual = x
        x_norm = self.norm1(x)

        if self.use_mamba:
            x_out = self.mamba(x_norm)
        else:
            x_out, _ = self.mamba(x_norm)

        x = residual + self.dropout(x_out)

        # ----------------------------------------------------
        # Feed Forward + Residual
        # ----------------------------------------------------
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


# ============================================================
# Frequency-Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    """
    沿频率轴进行建模的多头注意力模块。

    输入:
        x: [B * Tp, Fp, D]

    其中:
        B  : batch size
        Tp : time patch 数量
        Fp : frequency patch 数量
        D  : token embedding dimension

    对于每一个时间位置，模型沿频率方向计算注意力。
    """

    def __init__(
        self,
        dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} 必须能够被 num_heads={num_heads} 整除。"
            )

        # Attention 前的归一化
        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # FFN 前的归一化
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(
            dim=dim,
            dropout=dropout
        )

    def forward(self, x):
        """
        参数:
            x: [B * Tp, Fp, D]

        返回:
            x: [B * Tp, Fp, D]
        """

        # ----------------------------------------------------
        # Frequency Attention + Residual
        # ----------------------------------------------------
        residual = x
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False
        )

        x = residual + self.dropout(attn_out)

        # ----------------------------------------------------
        # Feed Forward + Residual
        # ----------------------------------------------------
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


# ============================================================
# Serial Time-Frequency Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
    """
    串联式时频编码器：

        Input
          ↓
        Time-Mamba
          ↓
        Frequency-Attention
          ↓
        Global Mean Pooling
          ↓
        Feature

    输入:
        x: [B, N, D]

    当前默认设置:
        N = 948
        D = 768

        948 = 12 × 79

        Fp = 12
        Tp = 79

    因此输入首先被恢复为:

        [B, 12, 79, 768]

    输出:
        feature: [B, 768]
    """

    def __init__(
        self,
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.token_dim = token_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            self.freq_patches * self.time_patches
        )

        # 输入 Token 归一化
        self.input_norm = nn.LayerNorm(token_dim)

        # ----------------------------------------------------
        # Time-Mamba blocks
        # ----------------------------------------------------
        self.time_blocks = nn.ModuleList([
            TimeMambaBlock(
                dim=token_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
            for _ in range(time_depth)
        ])

        # ----------------------------------------------------
        # Frequency-Attention blocks
        # ----------------------------------------------------
        self.freq_blocks = nn.ModuleList([
            FrequencyAttentionBlock(
                dim=token_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(freq_depth)
        ])

        # 最终输出归一化
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        """
        参数:
            x: [B, N, D]

        例如:
            x: [B, 948, 768]

        返回:
            feature: [B, D]
        """

        if x.ndim != 3:
            raise ValueError(
                "TimeFrequencyEncoder 的输入必须是三维张量 "
                "[B, N, D]，"
                f"但当前输入形状为 {tuple(x.shape)}。"
            )

        B, N, D = x.shape

        # ----------------------------------------------------
        # 检查 Token 数量
        # ----------------------------------------------------
        if N != self.num_tokens:
            raise ValueError(
                f"Token 数量不匹配：输入 N={N}，"
                f"但模型设置为 "
                f"freq_patches={self.freq_patches}，"
                f"time_patches={self.time_patches}，"
                f"要求 N={self.num_tokens}。"
            )

        # ----------------------------------------------------
        # 检查 Token 特征维度
        # ----------------------------------------------------
        if D != self.token_dim:
            raise ValueError(
                f"Token 维度不匹配：输入 D={D}，"
                f"但模型 token_dim={self.token_dim}。"
            )

        # ----------------------------------------------------
        # 输入归一化
        # [B, N, D]
        # ----------------------------------------------------
        x = self.input_norm(x)

        # ----------------------------------------------------
        # 恢复二维时频结构
        #
        # [B, N, D]
        #       ↓
        # [B, Fp, Tp, D]
        #
        # 当前默认:
        # [B, 948, 768]
        #       ↓
        # [B, 12, 79, 768]
        # ----------------------------------------------------
        x = x.reshape(
            B,
            self.freq_patches,
            self.time_patches,
            self.token_dim
        )

        # ====================================================
        # 第一阶段：Time-Mamba
        #
        # 对每一个频率 patch，
        # 单独沿时间轴 Tp 进行 Mamba 建模。
        # ====================================================
        time_x = x

        for block in self.time_blocks:

            # [B, Fp, Tp, D]
            #       ↓
            # [B * Fp, Tp, D]
            time_seq = time_x.reshape(
                B * self.freq_patches,
                self.time_patches,
                self.token_dim
            )

            # 沿时间轴进行 Mamba 建模
            time_seq = block(time_seq)

            # [B * Fp, Tp, D]
            #       ↓
            # [B, Fp, Tp, D]
            time_x = time_seq.reshape(
                B,
                self.freq_patches,
                self.time_patches,
                self.token_dim
            )

        # ====================================================
        # 第二阶段：Frequency-Attention
        #
        # 关键修改：
        # Frequency-Attention 不再使用原始输入 x，
        # 而是使用 Time-Mamba 的输出 time_x。
        #
        # 因此形成真正的串联：
        #
        # Time-Mamba → Frequency-Attention
        # ====================================================
        freq_x = time_x

        for block in self.freq_blocks:

            # ------------------------------------------------
            # 将频率维度移动到序列维度
            #
            # [B, Fp, Tp, D]
            #       ↓ permute
            # [B, Tp, Fp, D]
            # ------------------------------------------------
            freq_seq = freq_x.permute(
                0, 2, 1, 3
            ).contiguous()

            # ------------------------------------------------
            # 将 B 和 Tp 合并
            #
            # [B, Tp, Fp, D]
            #       ↓
            # [B * Tp, Fp, D]
            #
            # 此时每一条序列都是某一个时间位置上的
            # 完整频率序列。
            # ------------------------------------------------
            freq_seq = freq_seq.reshape(
                B * self.time_patches,
                self.freq_patches,
                self.token_dim
            )

            # 沿频率轴计算 Multi-Head Attention
            freq_seq = block(freq_seq)

            # ------------------------------------------------
            # 恢复:
            #
            # [B * Tp, Fp, D]
            #       ↓
            # [B, Tp, Fp, D]
            # ------------------------------------------------
            freq_seq = freq_seq.reshape(
                B,
                self.time_patches,
                self.freq_patches,
                self.token_dim
            )

            # ------------------------------------------------
            # 恢复为统一的时频排列:
            #
            # [B, Tp, Fp, D]
            #       ↓
            # [B, Fp, Tp, D]
            # ------------------------------------------------
            freq_x = freq_seq.permute(
                0, 2, 1, 3
            ).contiguous()

        # ====================================================
        # 第三阶段：全局平均池化
        #
        # freq_x 已经依次经过:
        #
        # Time-Mamba
        #      ↓
        # Frequency-Attention
        #
        # 因此不再需要与 time_x 进行拼接。
        # ====================================================

        # [B, Fp, Tp, D]
        #       ↓ mean over Fp and Tp
        # [B, D]
        feature = freq_x.mean(dim=(1, 2))

        # 输出归一化
        feature = self.output_norm(feature)

        return feature


# ============================================================
# 简单形状测试
# 直接运行 model.py 时才会执行
# ============================================================
if __name__ == "__main__":

    print(f"是否使用真实 Mamba: {HAS_MAMBA}")

    model = TimeFrequencyEncoder(
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    )

    dummy_input = torch.randn(
        2,
        948,
        768
    )

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print("输入形状:", dummy_input.shape)
    print("输出形状:", dummy_output.shape)