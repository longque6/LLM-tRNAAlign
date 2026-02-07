"""
attention_utils.py

该模块提供了一些注意力相关的辅助函数，主要包括：
  - utils_softmax: 封装 softmax 函数，支持 ONNX 导出时的特殊处理。
  - apply_sparse_mask: 用于将稀疏掩码应用到注意力 logits 上，
      例如在计算注意力分数时，将 padding 或不需要关注的位置置为 -∞ 以抑制其影响。
      
在多头注意力模块（如 MultiheadAttention）中，这些工具函数可以直接被导入使用，
从而减少重复代码，便于维护和扩展。
"""

import torch
import torch.nn.functional as F

def utils_softmax(x: torch.Tensor, dim: int, onnx_trace: bool = False) -> torch.Tensor:
    """
    封装的 softmax 函数，根据 onnx_trace 标志决定是否将输入转换为 float 类型，
    以支持 ONNX 导出。

    Args:
        x (Tensor): 输入张量。
        dim (int): softmax 应用的维度。
        onnx_trace (bool): 是否用于 ONNX 导出。若为 True，则先转换为 float。

    Returns:
        Tensor: softmax 后的结果。
    """
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

def apply_sparse_mask(attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor: # type: ignore
    """
    将稀疏掩码应用于注意力 logits。
    
    这里的实现为一个占位函数，直接返回原始 attn_weights。
    在实际应用中，可以扩展此函数，将不需要关注的区域（如 padding 位置）设为 -∞，
    以确保 softmax 后对应的注意力概率为 0。

    Args:
        attn_weights (Tensor): 注意力 logits，形状为 [batch_size * num_heads, tgt_len, src_len]。
        tgt_len (int): 目标序列长度。
        src_len (int): 源序列长度。
        bsz (int): 批量大小。

    Returns:
        Tensor: 应用掩码后的注意力 logits。
    """
    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights