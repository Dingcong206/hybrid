# mymodels/model.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


# ============================================================
# 1) AST Feature Extractor (内部选择 hidden_states[layer=12])
# ============================================================
class ASTFeatureExtractor(nn.Module):
    """
    在模型内部完成：
      - 调用 HuggingFace AST/Audio transformer
      - output_hidden_states=True
      - 取 hidden_states[layer_idx] 作为 token 序列

    forward 输入：
      ast_inputs: dict，直接传给 AutoModel(**ast_inputs)
    输出：
      tokens: (B, T, C)
    """

    def __init__(self, model_name: str, layer_idx: int = 12, freeze: bool = True):
        super().__init__()
        self.layer_idx = int(layer_idx)

        cfg = AutoConfig.from_pretrained(model_name)
        cfg.output_hidden_states = True  # 确保能返回 hidden_states

        self.ast = AutoModel.from_pretrained(model_name, config=cfg)

        if freeze:
            for p in self.ast.parameters():
                p.requires_grad = False

        # 推断输出维度（hidden size）
        out_dim = getattr(cfg, "hidden_size", None)
        if out_dim is None and hasattr(cfg, "hidden_sizes"):
            # 某些模型可能是 hidden_sizes 列表
            out_dim = cfg.hidden_sizes[-1]
        if out_dim is None:
            raise ValueError("无法从 config 推断 hidden size（out_dim），请检查模型类型。")

        self.out_dim = int(out_dim)

    def forward(self, ast_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ast_inputs: dict, e.g. {"input_values": ..., "attention_mask": ...} 具体键取决于你用的 processor
        return: tokens (B, T, C)
        """
        out = self.ast(**ast_inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states  # tuple length usually = num_layers + 1

        # 支持负数层索引
        idx = self.layer_idx if self.layer_idx >= 0 else (len(hs) + self.layer_idx)
        if idx < 0 or idx >= len(hs):
            raise IndexError(f"layer_idx={self.layer_idx} 越界：hidden_states 长度={len(hs)}")

        return hs[idx]  # (B,T,C)


# ============================================================
# 2) 基础组件：FFN / BiMamba / Attention
# ============================================================
class FFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiMambaBlock(nn.Module):
    """
    BiMamba: forward + backward (flip) + residual + FFN
    输入/输出: (B,T,D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("mamba_ssm 未安装：pip install mamba-ssm causal-conv1d")

        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        y_f = self.fwd(h)
        y_b = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        x = x + self.drop(y_f + y_b)

        x = x + self.ffn(self.ln2(x))
        return x


class AttnBlock(nn.Module):
    """
    用标准 TransformerEncoderLayer 做一个 attention block
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: (B,T) True=PAD
        return self.layer(x, src_key_padding_mask=key_padding_mask)


# ============================================================
# 3) 宏 Block：3 BiMamba + 1 Attention + 3 BiMamba
# ============================================================
class MacroBlock313(nn.Module):
    """
    一个宏 block 内部固定结构：
      BiMamba x3 -> Attention x1 -> BiMamba x3
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dropout: float = 0.2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()

        self.pre = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(3)
            ]
        )

        self.attn = AttnBlock(d_model=d_model, nhead=nhead, dropout=dropout, ffn_mult=ffn_mult)

        self.post = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(3)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for blk in self.pre:
            x = blk(x)

        x = self.attn(x, key_padding_mask=mask)

        for blk in self.post:
            x = blk(x)

        return x


# ============================================================
# 4) 最终模型：AST(layer=12) + (N个宏block) + pooling + head
# ============================================================
class HybridAST_MambaAttn_313(nn.Module):
    """
    满足你的要求：
      - layer=12 在模型内部选
      - 每个 block 内 3+1+3
      - 可堆叠 num_blocks 个这样的宏block

    forward 输入：
      ast_inputs: Dict[str, Tensor]
      mask: Optional[Tensor]  (B,T) True=PAD
           如果你不传 mask，会自动全 False（即全有效）
    输出：
      logits: (B, num_classes)
    """

    def __init__(
        self,
        ast_name: str,
        num_classes: int = 4,
        ast_layer_idx: int = 12,
        freeze_ast: bool = True,
        # hybrid backbone
        d_model: int = 256,
        num_blocks: int = 2,  # 多少个“宏 block”（每个宏block内部就是3+1+3）
        nhead: int = 8,
        dropout: float = 0.2,
        # mamba
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # ffn
        ffn_mult: int = 4,
        # pooling
        pooling: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        pooling = pooling.lower().strip()
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")

        self.pooling = pooling

        # 1) AST hidden layer selector
        self.ast_feat = ASTFeatureExtractor(
            model_name=ast_name,
            layer_idx=ast_layer_idx,
            freeze=freeze_ast,
        )
        in_dim = self.ast_feat.out_dim

        # 2) 投影到你的 d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.drop = nn.Dropout(dropout)

        # 3) 堆叠宏block，每个block内部固定 3+1+3
        self.blocks = nn.ModuleList(
            [
                MacroBlock313(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_feat_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def _pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        mask: (B,T) True=PAD
        return: (B,D)
        """
        if self.pooling == "cls":
            # 取第一个 token（如果你的 AST 第一个 token 是 CLS/summary token，这个就合理）
            return x[:, 0, :]

        # mean pooling（忽略 pad）
        valid = (~mask).float()  # (B,T)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, ast_inputs: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ast_inputs: HF AutoModel 的输入 dict
        mask: (B,T) True=PAD；若 None，则自动构造全 False
        """
        tokens = self.ast_feat(ast_inputs)  # (B,T,C)

        x = self.drop(self.in_proj(tokens))  # (B,T,d_model)

        B, T, _ = x.shape
        if mask is None:
            mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        for blk in self.blocks:
            x = blk(x, mask)

        pooled = self._pool(x, mask)  # (B,d_model)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 5) 兼容你之前训练脚本的 build_backbone（可选）
#    如果你还想用 build_backbone + classifier 的训练流程
# ============================================================
def build_backbone(
    # 这里给一个“兼容入口”，但本质返回的是一个能输出 pooled feat 的 backbone
    ast_name: str,
    ast_layer_idx: int = 12,
    freeze_ast: bool = True,
    d_model: int = 256,
    num_blocks: int = 2,
    nhead: int = 8,
    dropout: float = 0.2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    ffn_mult: int = 4,
    pooling: str = "mean",
):
    """
    返回一个 backbone：forward(ast_inputs, mask)->feat(B,d_model)
    """
    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = HybridAST_MambaAttn_313(
                ast_name=ast_name,
                num_classes=4,  # 这里无所谓，后面不走 classifier
                ast_layer_idx=ast_layer_idx,
                freeze_ast=freeze_ast,
                d_model=d_model,
                num_blocks=num_blocks,
                nhead=nhead,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ffn_mult=ffn_mult,
                pooling=pooling,
            )
            self.final_feat_dim = self.net.final_feat_dim

        def forward(self, ast_inputs: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
            # 复用内部流程，但拿 pooled 特征，不走分类头
            tokens = self.net.ast_feat(ast_inputs)
            x = self.net.drop(self.net.in_proj(tokens))
            B, T, _ = x.shape
            if mask is None:
                mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            for blk in self.net.blocks:
                x = blk(x, mask)
            feat = self.net._pool(x, mask)
            return feat

    return _Backbone()
