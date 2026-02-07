import random

# 随机选择其他碱基
def random_base():
    return random.choice(["A", "U", "G", "C"])

# 掩码处理，统一规则：80% 使用 [MASK] 替换，10% 随机替换为其他碱基，10% 保持原值
def apply_masking_rule(seq_list, mask_prob):
    masked_seq = seq_list.copy()
    label = [-100] * len(seq_list)  # Default label for unmasked tokens
    
    for i in range(len(seq_list)):
        if random.random() < mask_prob:  # Apply masking based on probability
            # 80%掩码为 [MASK]
            if random.random() < 0.8:
                masked_seq[i] = '[MASK]'
                label[i] = seq_list[i]  # Store the original token in the label
            # 10%随机替换为其他碱基
            elif random.random() < 0.1:
                masked_seq[i] = random_base()
                label[i] = seq_list[i]  # Store the original token in the label
            # 10%不做任何改变
            else:
                masked_seq[i] = seq_list[i]
                label[i] = seq_list[i]  # Store the original token in the label
                
    return ''.join(masked_seq), label

# 第一种方式：随机掩码，学习全部序列信息
def apply_random_mask(sequences, mask_prob=0.15):
    """
    Apply random masking to RNA sequences.
    Args:
        sequences (list of str): List of RNA sequences to mask.
        mask_prob (float): The probability of masking a token.
    
    Returns:
        masked_sequences: List of masked RNA sequences.
        labels: The original sequences used as labels for MLM.
    """
    masked_sequences = []
    labels = []
    for seq in sequences:
        seq_list = list(seq)
        masked_seq, label = apply_masking_rule(seq_list, mask_prob)
        masked_sequences.append(masked_seq)
        labels.append(label)
    
    return masked_sequences, labels

# 第二种方式：掩码二级结构中的 '.' 位置
def apply_dot_based_mask(sequences, structures, mask_prob=0.15):
    """
    Apply masking based on the secondary structure where '.' represents unpaired bases.
    Args:
        sequences (list of str): List of RNA sequences to mask.
        structures (list of str): List of secondary structure information corresponding to the sequences.
        mask_prob (float): The probability of masking a token.
    
    Returns:
        masked_sequences: List of masked RNA sequences.
        labels: The original sequences used as labels for MLM.
    """
    masked_sequences = []
    labels = []
    for seq, struct in zip(sequences, structures):
        seq_list = list(seq)
        struct_list = list(struct)
        
        masked_seq = seq_list.copy()
        label = [-100] * len(seq_list)

        for i in range(len(struct_list)):
            if struct_list[i] == '.' and random.random() < mask_prob:  # Apply masking based on structure
                masked_seq[i] = '[MASK]'
                label[i] = seq_list[i]
                
        masked_seq, label = apply_masking_rule(masked_seq, mask_prob)  # Apply masking rule for 80%/[MASK], 10% random replacement
        
        masked_sequences.append(masked_seq)
        labels.append(label)

    return masked_sequences, labels

# 第三种方式：掩码配对的碱基
def apply_pair_based_mask(sequences, structures, mask_prob=0.15):
    """
    Apply masking to paired bases based on the secondary structure marks '>' and '<'.
    Args:
        sequences (list of str): List of RNA sequences to mask.
        structures (list of str): List of secondary structure information corresponding to the sequences.
        mask_prob (float): The probability of masking a token.
    
    Returns:
        masked_sequences: List of masked RNA sequences.
        labels: The original sequences used as labels for MLM.
    """
    masked_sequences = []
    labels = []
    for seq, struct in zip(sequences, structures):
        seq_list = list(seq)
        struct_list = list(struct)
        
        masked_seq = seq_list.copy()
        label = [-100] * len(seq_list)

        # Iterate over structure to identify paired bases
        for i in range(len(struct_list)):
            if struct_list[i] in ['>', '<'] and random.random() < mask_prob:  # Apply masking to paired bases
                masked_seq[i] = '[MASK]'
                label[i] = seq_list[i]
                
        masked_seq, label = apply_masking_rule(masked_seq, mask_prob)  # Apply masking rule for 80%/[MASK], 10% random replacement
        
        masked_sequences.append(masked_seq)
        labels.append(label)

    return masked_sequences, labels

# 第四种方式：区域掩码
def apply_random_region_mask(sequences, regions, mask_prob=0.15):
    """
    Apply masking to three randomly selected regions of RNA sequences.
    Args:
        sequences (list of str): List of RNA sequences to mask.
        regions (list of dict): Region information specifying which parts of the sequence are masked.
        mask_prob (float): The probability of masking a token.
    
    Returns:
        masked_sequences: List of masked RNA sequences.
        labels: The original sequences used as labels for MLM.
    """
    masked_sequences = []
    labels = []

    for seq, region in zip(sequences, regions):
        seq_list = list(seq)
        masked_seq = seq_list.copy()
        label = [-100] * len(seq_list)  # -100 means this token is not used for loss calculation
        
        # Randomly select three regions to mask
        selected_regions = random.sample(list(region.items()), 3)
        
        for region_name, (start, end) in selected_regions:
            # Ensure the region is within bounds
            end = min(end, len(seq_list))  # Prevent out-of-bound error
            
            # Mask the selected region based on the mask_prob
            for i in range(start, end):
                if random.random() < mask_prob:  # Apply masking based on probability
                    masked_seq[i] = '[MASK]'
                    label[i] = seq_list[i]  # Store the original token in the label
        
        masked_seq, label = apply_masking_rule(masked_seq, mask_prob)  # Apply masking rule for 80%/[MASK], 10% random replacement
        
        masked_sequences.append(masked_seq)
        labels.append(label)

    return masked_sequences, labels

# 集成掩码函数：集成所有掩码策略，默认情况下，随机选择一个掩码策略
def apply_mask(sequences, structures=None, regions=None, mask_type="random", mask_prob=0.15):
    """
    Apply one of the mask strategies to RNA sequences. 
    Args:
        sequences (list of str): List of RNA sequences to mask.
        structures (list of str, optional): Secondary structure information.
        regions (list of dict, optional): Region information for region-based masking.
        mask_type (str): Type of mask to apply: "random", "dot", "pair", or "region".
        mask_prob (float): The probability of masking a token.
    
    Returns:
        masked_sequences: List of masked RNA sequences.
        labels: The original sequences used as labels for MLM.
    """
    if mask_type == "random":
        return apply_random_mask(sequences, mask_prob)
    elif mask_type == "dot":
        if structures is None:
            raise ValueError("Secondary structure information must be provided for dot-based masking.")
        return apply_dot_based_mask(sequences, structures, mask_prob)
    elif mask_type == "pair":
        if structures is None:
            raise ValueError("Secondary structure information must be provided for pair-based masking.")
        return apply_pair_based_mask(sequences, structures, mask_prob)
    elif mask_type == "region":
        if regions is None:
            raise ValueError("Region information must be provided for region-based masking.")
        return apply_random_region_mask(sequences, regions, mask_prob)
    else:
        raise ValueError("Invalid mask_type. Choose from 'random', 'dot', 'pair', or 'region'.")

#############################################
# 示例使用
#############################################

if __name__ == "__main__":
    sequences = [
        "GACCAGGUGGCCGAGUGGUUACGGUGAUGGAGUGCUAAUCCAUUGUGCUCUGCCUGCGUGGGUUCGAAUUCCAUACUCGUCG",
        "GGCUCGUUGGUCUAGGGGUAUGAUUCUCGCUUCGGGUGCGAGAGGUCCCGGGUUCAAAUCACGGACGAGCCC",
        "GGGAGCUUAGCUCAGCUGGGAGAGCAUUUGCCUUACAAGCAAGAGGUCACAGGUUCGAUCCCUGUAGCUCCCA",
    ]
    structures = [
        ">>>.>>...>>>>........<<<<.>>>>>.......<<<<<..............>>>>>.......<<<<<.<<.<<<.",
        ">>>>>>>..>>>.........<<<.>>>>>.......<<<<<.....>>>.>.......<.<<<<<<<<<<.",
        ">>>>>>>..>>>>........<<<<.>>>>>.......<<<<<.....>>>>>.......<<<<<<<<<<<<.",
    ]
    regions = [
        {'AA_Arm_5prime_Seq': (0, 8), 'D_Loop_Seq': (9, 25),'Anticodon_Arm_Seq': (26, 42), 'Variable_Loop_Seq': (43, 56), 'T_Arm_Seq': (57, 73), 'AA_Arm_3prime_Seq': (74, 81)},
        {'AA_Arm_5prime_Seq': (0, 8), 'D_Loop_Seq': (9, 24), 'Anticodon_Arm_Seq': (25, 41), 'Variable_Loop_Seq': (42, 46), 'T_Arm_Seq': (47, 63), 'AA_Arm_3prime_Seq': (64, 71)},
        {'AA_Arm_5prime_Seq': (0, 8), 'D_Loop_Seq': (9, 25), 'Anticodon_Arm_Seq': (26, 42), 'Variable_Loop_Seq': (43, 47), 'T_Arm_Seq': (48, 64), 'AA_Arm_3prime_Seq': (65, 72)},
    ]

    # 调用掩码函数
    masked_sequences, labels = apply_mask(sequences, structures=structures, regions=regions, mask_type="random", mask_prob=0.15)

    # 打印输出
    for seq, masked, label in zip(sequences, masked_sequences, labels):
        print(f"Original: {seq}")
        print(f"Masked: {masked}")
        print(f"Labels: {label}")
        print("-" * 80)