import pandas as pd
import torch
import torch.nn as nn
from typing import Dict
from AYLM.RNADataset import RNADataset, collate_fn

class RNATransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_segments: int,mask_token_id: int = 5):
        """
        RNATransformerEmbedding 实现了对 RNA 序列的多 region 输入做 Embedding 拼接，输出：
        1. token+segment embedding：形状 [B, total_seq_len, 2*d_model]
        2. mask+segment  embedding：形状 [B, total_seq_len, 2*d_model]
        3. flat_tokens           ：形状 [B, total_seq_len]（原始 token ID）
        4. flat_structures       ：形状 [B, total_seq_len]（原始 structure ID）
        5. flat_msk_tokens       ：形状 [B, total_seq_len]（被替换为 mask token 的 ID）
        6. mask_idx_list         ：长度 B 的 Python 列表，每项是该样本所有 mask 位置的索引列表

        Args:
            vocab_size (int):     token IDs 的最大值 + 1（包含 padding=0 和 mask_token_id）
            d_model (int):        单路 embedding 的维度
            n_segments (int):     segment ID（region ID）的类别数（通常是 region 数目 + 1，用于 padding）
            mask_token_id (int):  用于 Masked LM 的 mask token ID（默认 5，可选）
        """
        super().__init__()
        self.token_embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.mask_embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        # 保存 mask_token_id，避免硬编码
        self.mask_token_id = mask_token_id

    def forward(self,
                region_tokens: Dict[int, torch.Tensor],
                region_tokens_mask: Dict[int, torch.Tensor],
                segment_ids: Dict[int, torch.Tensor],
                region_structures: Dict[int, torch.Tensor]
               ):
        """
        前向计算。

        Args:
            region_tokens      (Dict[int, Tensor]): 每个 region 的原始 token IDs，Tensor 形状 [B, L_r]
            region_tokens_mask (Dict[int, Tensor]): 每个 region 的 mask 后 token IDs（mask→mask_token_id），[B, L_r]
            segment_ids        (Dict[int, Tensor]): 每个 region 的 segment/region IDs，[B, L_r]
            region_structures  (Dict[int, Tensor]): 每个 region 的结构标签 IDs，[B, L_r]

        Returns:
            tok_seg          (Tensor): 原 token + segment embedding, 形状 [B, ΣL_r, 2*d_model]
            msk_seg          (Tensor): mask token + segment embedding, 形状 [B, ΣL_r, 2*d_model]
            flat_tokens      (Tensor): 扁平化的原始 token IDs, 形状 [B, ΣL_r]
            flat_structures  (Tensor): 扁平化的结构标签 IDs, 形状 [B, ΣL_r]
            flat_msk_tokens  (Tensor): 扁平化的 mask 后 token IDs, 形状 [B, ΣL_r]
            mask_idx_list    (List[List[int]]):
                                     Python 列表长度为 B，每个子列表是该样本中所有等于 mask_token_id
                                     的位置索引，例如 [[3,17,25], [5,9], …]
        """
        tok_seg_list = []
        msk_seg_list = []
        flat_tokens_list    = []
        flat_structures_list= []
        flat_msk_tokens_list=[]

        # 按 region_id 的顺序拼
        for r in sorted(region_tokens.keys()):
            toks = region_tokens[r]         # [B, L_r]
            msks = region_tokens_mask[r]    # [B, L_r]
            segs = segment_ids[r]           # [B, L_r]
            strs = region_structures[r]     # [B, L_r]

            # lookup
            tok_emb = self.token_embedding(toks)   
            msk_emb = self.mask_embedding(msks)    

            # segment embedding：先 clamp 再 zero‑out pad 部分
            segs_clamped = segs.clamp(min=0)  
            seg_emb = self.segment_embedding(segs_clamped)
            seg_emb = seg_emb.masked_fill((segs == 0).unsqueeze(-1), 0.0) 

            # 拼接两种 embedding
            tok_seg_list.append(torch.cat([tok_emb, seg_emb], dim=-1))  
            msk_seg_list.append(torch.cat([msk_emb, seg_emb], dim=-1))  

            # 原始 ID
            flat_tokens_list.append(toks)
            flat_structures_list.append(strs)
            flat_msk_tokens_list.append(msks)

        # 把所有 region 在 seq 维度上接在一起
        tok_seg = torch.cat(tok_seg_list,        dim=1)  # [B, sum L_r, 2*d_model]
        msk_seg = torch.cat(msk_seg_list,        dim=1)  # [B, sum L_r, 2*d_model]
        flat_tokens     = torch.cat(flat_tokens_list,     dim=1)  # [B, sum L_r]
        flat_structures = torch.cat(flat_structures_list, dim=1)  # [B, sum L_r]
        flat_msk_tokens=torch.cat(flat_msk_tokens_list,dim=1)

        # 为每个样本收集所有 mask==5 的位置
        B, T = flat_msk_tokens.shape
        mask_idx_list = []
        for b in range(B):
            idxs = (flat_msk_tokens[b] == self.mask_token_id).nonzero(as_tuple=False).view(-1).tolist()
            mask_idx_list.append(idxs)

        return tok_seg, msk_seg, flat_tokens, flat_structures,flat_msk_tokens, mask_idx_list


if __name__ == "__main__":
    # 1) 准备数据
    df = pd.read_csv("all_dataset/train_set.csv")
    dataset = RNADataset(df)
    batch   = [dataset[i] for i in range(3)]
    coll    = collate_fn(batch)

    region_tokens      = coll['region_tokens']
    region_tokens_mask = coll['region_tokens_mask']
    segment_ids        = coll['segment_ids']
    region_structures  = coll['region_structures']

    # 确认一下 shape
    print("region_tokens[0]:",      region_tokens[0].shape)        # [B, L0]
    print("region_tokens:",      region_tokens)        
    print("region_tokens_mask[0]:", region_tokens_mask[0].shape)   # [B, L0]
    print("region_tokens_mask:", region_tokens_mask)  
    print("segment_ids[0]:",        segment_ids[0].shape)         # [B, L0]
    print("segment_ids:",        segment_ids)         # [B, L0]
    print("region_structures[0]:",  region_structures[0].shape)    # [B, L0]

    # 2) 创建 embedding 模型
    vocab_size = 10
    d_model    = 16
    n_segments = 7  # 通常是 6+1

    model = RNATransformerEmbedding(vocab_size, d_model, n_segments)

    # 3) 前向
    tok_seg, msk_seg, flat_tokens, flat_structs,flat_msk_tokens, mask_idx_list = model(
        region_tokens, region_tokens_mask, segment_ids, region_structures
    )

    # 4) 打印最终拼好的尺寸
    total_len = sum(region_tokens[r].size(1) for r in region_tokens)
    print("Final token+seg emb shape:", tok_seg.shape)       # [B, total_len, 2*d_model]
    print("Final mask+seg  emb shape:", msk_seg.shape)       # [B, total_len, 2*d_model]
    print("Final flat_msk_tokens shape:",flat_msk_tokens)
    print("Final flat_tokens shape:",   flat_tokens)   # [B, total_len]
    print("Final flat_structs shape:",  flat_structs)  # [B, total_len]
    print("Mask idx list:",            mask_idx_list)
