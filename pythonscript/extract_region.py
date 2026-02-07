# extract_region.py
from pythonscript.extract import (
    extract_amino_acid_arm,
    extract_d_loop,
    extract_anticodon_arm,
    extract_variable_loop,
    extract_t_arm,
)

def extract_region(sequence: str, structure: str, region: str) -> str:
    """
    根据输入的区域名称，从给定的序列和二级结构中提取相应区域的序列。

    参数:
        sequence (str): tRNA 的一级序列。
        structure (str): tRNA 的二级结构标识（例如用 '>', '<', '.' 表示）。
        region (str): 要提取的区域名称，可以是以下之一：
            - "氨基酸臂a" 或 "amino_acid_arm_a"
            - "氨基酸臂b" 或 "amino_acid_arm_b"
            - "D环" 或 "D_loop"
            - "反密码子臂" 或 "anticodon_arm"
            - "可变环" 或 "variable_loop"
            - "T臂" 或 "T_arm"

    返回:
        str: 提取的区域序列；如果提取失败则返回 None。
    """
    if region in ["氨基酸臂a", "amino_acid_arm_a"]:
        result = extract_amino_acid_arm(sequence, structure)
        if result:
            return result[0]  # 返回氨基酸臂a（5′端）
        else:
            return None
    elif region in ["氨基酸臂b", "amino_acid_arm_b"]:
        result = extract_amino_acid_arm(sequence, structure)
        if result:
            return result[1]  # 返回氨基酸臂b（3′端）
        else:
            return None
    elif region in ["D环", "D_loop"]:
        return extract_d_loop(sequence, structure)
    elif region in ["反密码子臂", "anticodon_arm"]:
        return extract_anticodon_arm(sequence, structure)
    elif region in ["可变环", "variable_loop"]:
        return extract_variable_loop(sequence, structure)
    elif region in ["T臂", "T_arm"]:
        return extract_t_arm(sequence, structure)
    else:
        print(f"未识别的区域名称: {region}")
        return None

# 示例调用
if __name__ == '__main__':
    sequence = "ATGCGCGTAGCTCAGCTGGTAGAGCGGCGGTCTCCAAAACCGCAGGtCGTCGGATCGAAGCCAACCGCGC"
    structure = "..>>>>>..>>>>........<<<<.>>>>>.......<<<<<.....>>.>>.......<<.<<<<<<<"
    
    # 例如提取氨基酸臂a的序列
    region_to_extract = "氨基酸臂a"  # 或者 "amino_acid_arm_a"
    extracted_seq = extract_region(sequence, structure, region_to_extract)
    
    if extracted_seq:
        print(f"{region_to_extract}的序列: {extracted_seq}")
    else:
        print(f"{region_to_extract}提取失败。")