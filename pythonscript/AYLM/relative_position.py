"""
relative_position.py

该模块封装了相对位置编码的实现，主要用于 Transformer 的多头注意力中，
以捕捉序列中 token 间的相对位置信息，尤其适用于具有明显局部结构的序列（如 tRNA）。

包含内容：
  - RelativePositionEmbedding 类：根据最大相对位置和头维度生成相对位置嵌入，
      输入序列长度，输出形状为 [seq_len, seq_len, head_dim] 的编码矩阵。
  - compute_relative_position_scores 函数：利用 Query 与相对位置编码计算相对位置得分，
      返回形状为 [B, heads, seq_len, seq_len] 的得分矩阵。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码模块。
    
    Args:
        max_relative_positions (int): 最大相对位置。输出嵌入的词汇表大小为 2*max_relative_positions + 1。
        head_dim (int): 每个注意力头的维度。
    
    Forward:
        输入：当前序列长度 (int)
        输出：相对位置编码矩阵，形状为 [seq_len, seq_len, head_dim]
    """
    def __init__(self, max_relative_positions: int, head_dim: int):
        super(RelativePositionEmbedding, self).__init__()
        self.max_relative_positions = max_relative_positions
        vocab_size = 2 * max_relative_positions + 1
        self.embedding = nn.Embedding(vocab_size, head_dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        # 构造 [seq_len, seq_len] 的相对位置矩阵
        range_vec = torch.arange(seq_len, device=self.embedding.weight.device)
        # shape: [seq_len, 1] 和 [1, seq_len]
        rel_pos_mat = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)
        # 限制范围在 [-max_relative_positions, max_relative_positions]
        rel_pos_mat.clamp_(-self.max_relative_positions, self.max_relative_positions)
        # 将负数 shift 到 [0, 2*max_relative_positions]
        rel_pos_mat = rel_pos_mat + self.max_relative_positions
        # 通过嵌入层得到相对位置编码，形状为 [seq_len, seq_len, head_dim]
        rel_emb = self.embedding(rel_pos_mat)
        return rel_emb

def compute_relative_position_scores(Q_4d: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
    """
    计算相对位置得分：
      对于每个注意力头中的 Query，计算其与相对位置编码之间的点积得分。
    
    Args:
        Q_4d (Tensor): 重塑后的 Query，形状为 [B, heads, seq_len, head_dim]
        rel_emb (Tensor): 相对位置编码矩阵，形状为 [seq_len, seq_len, head_dim]
    
    Returns:
        Tensor: 相对位置得分矩阵，形状为 [B, heads, seq_len, seq_len]
    
    例如，对于一个 Query 中第 i 个 token 的向量 Q_4d(b, h, i, :)，
    得到与位置 j 对应的相对位置编码 rel_emb[i, j, :] 后，
    得分为 Q_4d(b, h, i, :) 与 rel_emb[i, j, :] 的内积。
    """
    # 使用 einsum 实现批量内的点积：'bhid,ijd -> bhij'
    rel_scores = torch.einsum('bhid,ijd->bhij', Q_4d, rel_emb)
    return rel_scores