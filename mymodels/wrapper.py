import torch
import torch.nn as nn

class SSA4ClassWrapper(nn.Module):
    """
    自适应版本：自动推断特征维度，避免你现在的报错
    """

    def __init__(self, base_model: nn.Module, feat_dim=None, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.base = base_model
        self.feat_dim = feat_dim  # 可能是 None

        # 先占位，真正的 head 在第一次 forward 时创建
        self.head = None
        self.num_classes = num_classes
        self.dropout = dropout

    def _build_head(self, feat_dim):
        """第一次 forward 时，根据真实特征维度构建分类头"""
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(self.dropout),
            nn.Linear(feat_dim, self.num_classes)
        )

    def forward(self, x, mask=None):
        out = self.base(x, mask) if mask is not None else self.base(x)

        # 兼容 tuple/list 输出
        if isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            feat = out

        # 🚨 如果 base 只输出 (B,) —— 需要改模型
        if feat.dim() == 1:
            raise RuntimeError(
                "❌ 你的 SSA_Model_HeARTokens 当前输出是 (B,) 二分类 logit，\n"
                "👉 必须改成输出 (B, D) 的文件级特征（embedding），否则无法做 4 类。"
            )

        # feat 应该是 (B, D)
        B, D = feat.shape

        # 第一次 forward 时自动创建 head
        if self.head is None:
            self._build_head(D)

        logits4 = self.head(feat)  # (B, 4)
        return logits4

