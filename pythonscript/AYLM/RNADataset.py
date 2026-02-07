# AYLM/RNADataset.py
import random
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from AYLM.mask import apply_mask  

class RNADataset(Dataset):
    def __init__(self, dataframe, mask_prob=0.15):
        """
        Args:
            dataframe (pd.DataFrame): 包含所有RNA序列和二级结构的DataFrame
            mask_prob (float): 掩码的概率
        """
        self.selected_columns = [
            'SecondaryStructure', 'StructFileSequence',
            'AA_Arm_5prime_Seq', 'AA_Arm_5prime_Struct',
            'AA_Arm_3prime_Seq', 'AA_Arm_3prime_Struct', 
            'D_Loop_Seq', 'D_Loop_Struct', 'Anticodon_Arm_Seq', 
            'Anticodon_Arm_Struct', 'Variable_Loop_Seq', 
            'Variable_Loop_Struct', 'T_Arm_Seq', 'T_Arm_Struct'
        ]
        self.region_columns = [
            'AA_Arm_5prime_Seq', 'D_Loop_Seq', 'Anticodon_Arm_Seq', 
            'Variable_Loop_Seq', 'T_Arm_Seq', 'AA_Arm_3prime_Seq'
        ]
        self.dataframe = dataframe
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # 提取所有序列和结构
        seqs, structs = [], []
        for col in self.selected_columns:
            if "Seq" in col:
                seqs.append(row[col])
            else:
                structs.append(row[col])

        # 计算 regions
        regions = self.get_regions(row)

        # apply_mask 返回掩码后的序列列表 + labels（不用这里）
        masked_seqs, _ = apply_mask(seqs, structures=structs, regions=regions, mask_prob=self.mask_prob)
        masked_full = ''.join(masked_seqs)

        # 按 region 划分原始/掩码/结构
        sf_seq_region = self.split_by_region(row['StructFileSequence'], regions)
        m_seq_region  = self.split_by_region(masked_full, regions)
        sf_str_region = self.split_by_region(row['SecondaryStructure'], regions)

        return {
            'StructFileSequenceRegion': sf_seq_region,
            'MaskedSequenceRegion':     m_seq_region,
            'StructFileStructureRegion': sf_str_region,
            'regions': regions
        }

    def get_regions(self, row):
        regions = {}
        cur = 0
        for col in self.region_columns:
            L = len(row[col])
            regions[col] = (cur, cur + L - 1)
            cur += L
        return regions

    def split_by_region(self, full_seq, regions):
        full_seq = full_seq.replace('[MASK]', 'M')
        out = {}
        for col, (s, e) in regions.items():
            out[col] = full_seq[s:e+1]
        return out


def collate_fn(batch):
    """
    batch: list of dicts, each contains three region‐dicts + regions info.
    返回：
      'region_tokens': dict[r] = LongTensor[B, L_r]
      'region_tokens_mask': dict[r] = LongTensor[B, L_r]
      'region_structures': dict[r] = LongTensor[B, L_r]
      'segment_ids': dict[r] = LongTensor[B, L_r] （区域 id 从1开始，pad=0）
    """
    B = len(batch)
    region_names = list(batch[0]['regions'].keys())
    R = len(region_names)

    # 临时存放各样本的 list
    rtok, rmask, rstr, rseg = [
        {r: [] for r in range(R)} for _ in range(4)
    ]

    # 简单映射
    def tokenize(seq, struct=False):
        m = {'pad':0,'A':1,'C':2,'G':3,'T':4,'U':4,'M':5}
        if struct:
            m2 = {'.':1,'<':2,'>':3}
            return [m2.get(c, 0) for c in seq]
        else:
            return [m.get(c, 0) for c in seq]

    # 先 collect
    for sample in batch:
        for r, name in enumerate(region_names):
            s_tok = tokenize(sample['StructFileSequenceRegion'][name])
            s_msk = tokenize(sample['MaskedSequenceRegion'][name])
            s_str = tokenize(sample['StructFileStructureRegion'][name], struct=True)
            rtok[r].append(torch.tensor(s_tok, dtype=torch.long))
            rmask[r].append(torch.tensor(s_msk, dtype=torch.long))
            rstr[r].append(torch.tensor(s_str, dtype=torch.long))
            # 区域从1开始编码
            rseg[r].append(torch.full((len(s_tok),), r+1, dtype=torch.long))

    # 再 pad
    out = {'region_tokens':{}, 'region_tokens_mask':{}, 'region_structures':{}, 'segment_ids':{}}
    for r in range(R):
        out['region_tokens'][r]      = pad_sequence(rtok[r],   batch_first=True, padding_value=0)
        out['region_tokens_mask'][r] = pad_sequence(rmask[r],  batch_first=True, padding_value=0)
        out['region_structures'][r]  = pad_sequence(rstr[r],   batch_first=True, padding_value=0)
        out['segment_ids'][r]        = pad_sequence(rseg[r],   batch_first=True, padding_value=0)

    return out


# 测试函数：从CSV文件加载数据并验证RNADataset的功能
# 运行测试
if __name__ == "__main__":
    # 加载CSV文件
    file_path = "all_dataset/train_set.csv"  # 请根据实际路径修改
    df = pd.read_csv(file_path)

    # selected_columns 和 region_columns 的定义

    # 创建RNADataset实例
    dataset = RNADataset(df)

    # 以批量形式进行处理
    batch = [dataset[i] for i in range(3)]  # 获取前三个样本
    collate_sample = collate_fn(batch)  # 调用collate_fn进行批处理

    # 打印输出
    print(f"Region Tokens: {collate_sample['region_tokens']}")
    print(f"Region Tokens Mask: {collate_sample['region_tokens_mask']}")
    print(f"Region Structures: {collate_sample['region_structures']}")
    print(f"Segment IDs: {collate_sample['segment_ids']}")