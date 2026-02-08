import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# ============================================================
# 0) RMSNorm 兼容
# ============================================================
def _rmsnorm(dim: int):
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    else:
        return CustomRMSNorm(dim)

class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


# ============================================================
# 1) 位置编码
# ============================================================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device="cpu"):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ============================================================
# 2) BiMambaBlock
# ============================================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("请安装 mamba-ssm: pip install mamba-ssm causal-conv1d")

        self.ln1 = _rmsnorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = _rmsnorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# 3) AttentionBlock
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.ln = _rmsnorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_n = self.ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        return x + self.drop(x_a)


# ============================================================
# 4) ICBHI Pooling (Attn + Mean)
# ============================================================
class ICBHI_Pooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask=None):
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), neg_inf)

        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)

        if mask is not None:
            x_valid = x.masked_fill(mask.unsqueeze(-1), 0.0)
            denom = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            feat_mean = x_valid.sum(dim=1) / denom
        else:
            feat_mean = x.mean(dim=1)

        return feat_weighted + feat_mean


# ============================================================
# 5) Stage：✅ 3 BiMamba + 1 Attention + 3 BiMamba
# ============================================================
class Stage3M1A3M(nn.Module):
    """
    一个 stage 内部结构：
      BiMamba -> BiMamba -> BiMamba -> Attention -> BiMamba -> BiMamba -> BiMamba
    """
    def __init__(self, d_model, nhead=8, dropout=0.3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.pre = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])
        self.attn = AttentionBlock(d_model, nhead=nhead, dropout=dropout)
        self.post = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

    def forward(self, x, mask=None):
        for blk in self.pre:
            x = blk(x)
        x = self.attn(x, mask=mask)
        for blk in self.post:
            x = blk(x)
        return x


# ============================================================
# 6) fbank -> AST patch projection -> tokens
# ============================================================
class ASTPatchProjection(nn.Module):
    """
    输入：fbank (B, 798, 128)
    输出：tokens (B, N, 768)
    """
    def __init__(
        self,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=True,
    ):
        super().__init__()
        from transformers import ASTModel
        ast = ASTModel.from_pretrained(ast_model_name, local_files_only=local_files_only)

        self.proj = ast.embeddings.patch_embeddings.projection

        # 默认冻结整个 AST
        for p in ast.parameters():
            p.requires_grad = False

        # 只解冻 projection
        if unfreeze_projection:
            for p in self.proj.parameters():
                p.requires_grad = True

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        fbank = fbank.float()
        x = fbank.transpose(1, 2).unsqueeze(1)    # (B,1,128,798)
        y = self.proj(x)                          # (B,768,F',T')
        tokens = y.flatten(2).transpose(1, 2)     # (B,N,768)
        return tokens


# ============================================================
# 7) 主模型：fbank -> AST proj tokens -> SSA
# ============================================================
class SSA_Model_FbankToSSA(nn.Module):
    def __init__(
        self,
        in_dim=768,
        d_model=512,
        n_layers=2,
        nhead=8,
        dropout=0.3,
        max_len=1024,
        num_classes=1,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=True,
    ):
        super().__init__()

        self.ast_proj = ASTPatchProjection(
            ast_model_name=ast_model_name,
            local_files_only=local_files_only,
            unfreeze_projection=unfreeze_projection,
        )

        self.input_proj = nn.Sequential(
            _rmsnorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.SiLU()
        )
        self.d_model = d_model
        self.num_classes = num_classes

        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        self.front_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=1, bias=False),
            nn.SiLU(),
        )
        self.front_ln = _rmsnorm(d_model)

        # ✅ 核心：换成 3M + A + 3M
        self.stages = nn.ModuleList([
            Stage3M1A3M(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, fbank, mask=None, return_feature=False):
        x = self.ast_proj(fbank)                  # (B,N,768)
        x = self.input_proj(x)                    # (B,N,d_model)

        Tt = x.shape[1]
        x = x + self.pe[:, :Tt, :].to(x.device)
        x = self.pos_drop(x)

        y = self.front_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.front_ln(y)

        for stage in self.stages:
            x = stage(x, mask=mask)

        x = self.final_norm(x)
        file_feature = self.pool(x, mask=mask)

        if return_feature:
            return file_feature, x

        return self.classifier(file_feature)


# ============================================================
# 8) Backbone 包装类
# ============================================================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_FbankToSSA):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, fbank, mask=None):
        feat, _ = self.ssa(fbank, mask=mask, return_feature=True)
        return feat


# ============================================================
# 9) build_model：返回 backbone
# ============================================================
def build_model(
    d_model=512,
    n_layers=2,
    nhead=8,
    num_classes=1,
    ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    local_files_only=False,
    unfreeze_projection=True,
):
    ssa = SSA_Model_FbankToSSA(
        in_dim=768,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        num_classes=num_classes,
        ast_model_name=ast_model_name,
        local_files_only=local_files_only,
        unfreeze_projection=unfreeze_projection,
    )
    backbone = SSA_Backbone(ssa)
    params = sum(p.numel() for p in backbone.parameters())
    print(f"Structure: fbank->AST(proj trainable={unfreeze_projection})->[3×BiMamba + Attn + 3×BiMamba]×{n_layers}")
    print(f"Total Params: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone


# ============================================================
# ✅ 兼容别名
# ============================================================
SSA_Model = SSA_Model_FbankToSSA
build_backbone = build_model
