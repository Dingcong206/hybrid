import torch
import torch.nn as nn


class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes=1, d_model=128, n_layers=2, freq_bins=128, patch_time=4):
        super().__init__()
        # 1. 前端：使用和你 VimA 一样的声学条带卷积，保证输入特征一致
        self.proj = nn.Conv2d(
            1, d_model,
            kernel_size=(freq_bins, patch_time),
            stride=(freq_bins, patch_time)
        )
        self.norm = nn.LayerNorm(d_model)

        # 2. 核心：双向 LSTM
        # batch_first=True 对应输入 (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0
        )

        # 3. 后端：分类头
        # 因为是双向，隐藏层维度会翻倍 (d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 128, 1024]

        # 提取条带特征
        x = self.proj(x)  # -> [B, d_model, 1, L]
        x = x.flatten(2).transpose(1, 2)  # -> [B, L, d_model]
        x = self.norm(x)

        # 进入 LSTM
        # lstm_out: [B, L, d_model * 2]
        lstm_out, _ = self.lstm(x)

        # 论文常用策略：取最后一个时间步，或者全局平均池化
        # 这里使用全局平均池化，对呼吸音这类长时特征更稳健
        out = torch.mean(lstm_out, dim=1)

        # 分类
        return self.head(out).squeeze(-1)  # -> [B]