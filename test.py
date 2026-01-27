import torch
import torch.nn as nn


class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes=1, d_model=128, n_layers=2, freq_bins=128, patch_time=4):
        super().__init__()
        # 1. 前端：声学条带卷积 (Stem) - 保持与 VimA 一致
        # 将频谱切成条带，输入维度与 Mamba 版本完全对齐
        self.proj = nn.Conv2d(
            1, d_model,
            kernel_size=(freq_bins, patch_time),
            stride=(freq_bins, patch_time)
        )
        self.norm = nn.LayerNorm(d_model)

        # 2. 核心：双向 LSTM (Bi-LSTM)
        # bidirectional=True 会让 hidden_size 翻倍
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0
        )

        # 3. 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 双向所以是 d_model * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 128, 1024]

        # 提取条带特征 (Acoustic Strip Feature)
        x = self.proj(x)  # -> [B, d_model, 1, L]
        x = x.flatten(2).transpose(1, 2)  # -> [B, L, d_model]
        x = self.norm(x)

        # 序列建模
        # lstm_out: [B, L, d_model * 2]
        lstm_out, _ = self.lstm(x)

        # 全局池化 (Global Mean Pooling)
        # 相比只取最后一个 hidden state，全局池化对呼吸周期的不固定噪声更鲁棒
        out = torch.mean(lstm_out, dim=1)

        return self.head(out).squeeze(-1)  # -> [B]