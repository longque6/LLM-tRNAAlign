"""
modules.py

本文件实现了 Transformer Encoder 的基本模块，包括层归一化、残差连接和前馈网络等组件。

包含以下部分：
  - LayerNorm：简单实现的层归一化模块
  - NormalizedResidualBlock：封装残差连接和层归一化
  - FeedForwardNetwork：前馈神经网络（FFN）
  - TransformerEncoderLayer：包含多头自注意力、残差连接、层归一化和 FFN 的完整层
  - TransformerEncoder：堆叠多个 TransformerEncoderLayer 构成完整的编码器
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 引入多头自注意力模块
from AYLM.modeling_rope_utils import PretrainedConfig
from AYLM.region_multihead_attention import MultiheadAttention


class LayerNorm(nn.Module):
    """
    简单的层归一化模块
    """
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        # x: [*, hidden_size]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class NormalizedResidualBlock(nn.Module):
    """
    封装残差连接和层归一化的模块。  
    给定一个子层 layer（例如多头注意力或前馈网络），
    先对输入进行层归一化，然后输入子层，最后将结果与输入相加并加上 dropout。
    如果子层返回 tuple，则保留 tuple 中的所有元素，将第一个元素做残差连接，
    并将其余元素原样返回。
    """
    def __init__(self, layer: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer   = layer
        self.norm    = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        # x: [*, hidden_size]
        residual = x
        x_norm   = self.norm(x)
        out = self.layer(x_norm, *args, **kwargs)

        if isinstance(out, tuple):
            # 如果子层返回 (primary_out, aux1, aux2, ...)
            primary, *aux = out
            primary = self.dropout(primary)
            return (residual + primary, *aux)
        else:
            # 仅返回主输出
            primary = self.dropout(out)
            return residual + primary

class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network, FFN)
    通常包含两个全连接层和一个非线性激活函数（如 GELU），以及 dropout。
    """
    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.act = nn.GELU()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 层：
      1. 多头自注意力（Pre-Norm + Attention + Dropout + Residual，返回 attn_weights）
      2. 前馈网络 (Pre-Norm + FFN + Dropout + Residual)
    输入：x [seq_len, batch_size, hidden_size]
    输出： (x, attn_weights)
    """
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        add_bias_kv: bool = True,
        use_region_attention: bool = False,
        rope_config: Optional[PretrainedConfig] = None,
        max_position: int = 256, 
    ):
        super().__init__()
        # —— Attention 部分 —— 
        self.self_attn_norm = LayerNorm(hidden_size)
        self.self_attn      = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            region_attention=use_region_attention,
            max_position=max_position, 
            rope_config=rope_config
        )
        self.attn_dropout   = nn.Dropout(dropout)

        # —— FFN 部分 —— 
        self.ffn_block = NormalizedResidualBlock(
            layer=FeedForwardNetwork(hidden_size, ffn_hidden_size, dropout),
            hidden_size=hidden_size,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        # 1) 多头自注意力子层
        residual = x
        x_norm   = self.self_attn_norm(x)
        # 注意这里正确地把 x_norm 同时当作 query/key/value
        attn_output, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        x = residual + self.attn_dropout(attn_output)

        # 2) 前馈网络子层（直接用通用的 NormalizedResidualBlock）
        x = self.ffn_block(x)

        # 返回编码后输出和注意力权重
        return x, attn_weights
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder，由多个 TransformerEncoderLayer 堆叠而成。
    输入形状： [seq_len, batch_size, hidden_size]
    输出形状： [seq_len, batch_size, hidden_size]
    """
    def __init__(self, num_layers: int, hidden_size: int, ffn_hidden_size: int, num_heads: int, dropout: float = 0.1,
                 add_bias_kv: bool = True, use_region_attention: bool = False, rope_config: Optional[PretrainedConfig] = None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, ffn_hidden_size, num_heads, dropout, add_bias_kv, use_region_attention,rope_config)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(hidden_size)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask, key_padding_mask)
            attn_weights_list.append(attn_weights)
        x = self.final_norm(x)
        return x, attn_weights_list

if __name__ == "__main__":
    import random
    import torch
    from AYLM.modeling_rope_utils import PretrainedConfig
    from AYLM.modules import LayerNorm, FeedForwardNetwork, NormalizedResidualBlock, TransformerEncoderLayer, TransformerEncoder

    # 固定随机种子，保证可复现
    random.seed(42)
    torch.manual_seed(42)

    # 超参数设定
    seq_len     = 10
    batch_size  = 2
    hidden_size = 16
    ffn_hidden  = 64
    num_heads   = 4
    num_layers  = 3
    dropout     = 0.1

    print("=== 超参数 ===")
    print(f"序列长度 (seq_len): {seq_len}")
    print(f"批大小 (batch_size): {batch_size}")
    print(f"隐藏维度 (hidden_size): {hidden_size}")
    print(f"FFN 隐藏维度: {ffn_hidden}")
    print(f"注意力头数: {num_heads}")
    print(f"Encoder 层数: {num_layers}")
    print(f"Dropout: {dropout}\n")

    # —— 定义一个动态 RoPE 的配置 —— 
    # 注意不要传 rope_type 进 __init__，而是在 rope_scaling 里指定，并在构造后手动覆盖
    rope_config = PretrainedConfig(
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        head_dim=hidden_size // num_heads,       # e.g. 16//4 = 4
        max_position_embeddings=256,             # 初始 max_position
        rope_scaling={"rope_type": "dynamic", "factor": 1.5},
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
    )
    # 手动覆盖，让它真正拿到 dynamic
    rope_config.rope_type = "dynamic"

    # 1) 测试 LayerNorm
    x = torch.randn(batch_size, seq_len, hidden_size)
    ln = LayerNorm(hidden_size)
    x_ln = ln(x)
    print(">>> LayerNorm 测试")
    print(" 输入形状:", x.shape)
    print(" 输出形状:", x_ln.shape)
    print(" 输出均值:", x_ln.mean().item(), "输出标准差:", x_ln.std().item(), "\n")

    # 2) 测试 FeedForwardNetwork
    ffn = FeedForwardNetwork(hidden_size, ffn_hidden, dropout)
    y_ffn = ffn(x)
    print(">>> FeedForwardNetwork 测试")
    print(" 输入形状:", x.shape)
    print(" 输出形状:", y_ffn.shape, "\n")

    # 3) 测试 NormalizedResidualBlock 封装 FFN
    block_ffn = NormalizedResidualBlock(ffn, hidden_size, dropout)
    y_block_ffn = block_ffn(x)
    print(">>> NormalizedResidualBlock（FFN）测试")
    print(" 输入形状:", x.shape)
    print(" 输出形状:", y_block_ffn.shape, "\n")

    # 4) 测试单层 TransformerEncoderLayer（启用动态 RoPE）
    L, B, H = seq_len, batch_size, hidden_size
    x_seq = torch.randn(L, B, H)
    short_max_pos = seq_len
    layer = TransformerEncoderLayer(
        hidden_size, ffn_hidden, num_heads, dropout,
        add_bias_kv=True,
        use_region_attention=False,
        rope_config=rope_config,   # ← 传入动态 RoPE 配置
        max_position=short_max_pos, 
    )
    out_layer, attn_weights = layer(x_seq)
    print(">>> TransformerEncoderLayer（动态 RoPE）测试")
    print(" 输入形状:", x_seq.shape)
    print(" 输出形状:", out_layer.shape)
    print(" 注意力权重形状:", attn_weights.shape, "\n")

    # 5) 测试多层 TransformerEncoder（启用动态 RoPE）
    encoder = TransformerEncoder(
        num_layers,
        hidden_size,
        ffn_hidden,
        num_heads,
        dropout,
        add_bias_kv=True,
        use_region_attention=False,
        rope_config=rope_config,   # ← 传入动态 RoPE 配置
    )
    out_enc, attn_list = encoder(x_seq)
    print(">>> TransformerEncoder（动态 RoPE）测试")
    print(" 编码层数:", num_layers)
    print(" 输出形状:", out_enc.shape)
    print(" 注意力列表长度:", len(attn_list))
    for idx, aw in enumerate(attn_list):
        print(f"  第 {idx+1} 层注意力权重形状: {aw.shape}")
    print("\n所有测试完成，动态 RoPE 已正确启用并通过验证！")

    # 验证动态 RoPE 扩容逻辑
    long_seq_len = 1024
    x_long = torch.randn(long_seq_len, batch_size, hidden_size)
    _ = layer(x_long)  # forward

    # 只检查 max_seq_len_cached
    cached = layer.self_attn.max_seq_len_cached
    print(f"max_seq_len_cached (expect == {long_seq_len}): {cached}")
    assert cached == long_seq_len, "max_seq_len_cached 没有更新！"
    print("✔ 动态 RoPE 扩容成功")