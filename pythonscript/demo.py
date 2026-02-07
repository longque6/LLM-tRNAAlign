#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块: align_with_dp_module
功能: 提供 _align_with_dp 函数，用本地 DP（PairwiseAligner）对齐并生成编号和序列。
"""
from typing import List, Dict
from alignment_score import alignment_score_and_str


def validate_alignment(target_seq: str, aligned_seq) -> bool:
    """
    验证对齐结果是否仅包含 A/U/G/C 且保留了与目标序列完全相同的碱基顺序。
    """
    valid_bases = set("AUGC")
    target_bases = "".join(ch for ch in target_seq if ch in valid_bases)
    if isinstance(aligned_seq, (list, tuple)):
        aligned_str = "".join(aligned_seq)
    else:
        aligned_str = aligned_seq
    aligned_bases = "".join(ch for ch in aligned_str if ch in valid_bases)
    return target_bases == aligned_bases


def _align_with_dp(
    template_num: List[str],
    template_seq: str,
    target_seq: str,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5
) -> Dict[str, List[str]]:
    """
    使用 PairwiseAligner 做全局对齐，并生成 aligned_numbering 与 aligned_sequence。
    - 插入(insertion)用“前一编号+iN”后缀。
    - 输出包含所有模板编号，未对齐或占位编号对应 '-'。
    """
    print("[Info] 开始 _align_with_dp")
    print(f"[Info] 模板编号列表: {template_num}")
    print(f"[Info] 模板序列: {template_seq}")
    print(f"[Info] 目标序列: {target_seq}")

    # 1) DP 对齐
    score, flat = alignment_score_and_str(target_seq, template_seq,
                                          match, mismatch, gap_open, gap_extend)
    print(f"[Info] 对齐得分: {score}")
    print("[Debug] flat alignment:")
    print(flat)

    # 2) 提取对齐行
    flat_target = ''  # DP 输出 'target' 行
    flat_template = ''  # DP 输出 'query' 行
    for line in flat.splitlines():
        parts = line.split()
        if line.startswith('target'):
            flat_target = parts[1]
        elif line.startswith('query'):
            flat_template = parts[1]
    print(f"[Debug] flat_template: {flat_template}")
    print(f"[Debug] flat_target  : {flat_target}")

    # 3) 逐列对齐
    aligned_nums: List[str] = []
    aligned_seqs: List[str] = []
    idx_template = 0
    last_num: str = None
    ins_count = 1

    for i, (templ_char, targ_char) in enumerate(zip(flat_template, flat_target), start=1):
        print(f"[Debug] 列 {i}: 模板('{templ_char}') 目标('{targ_char}')")
        # 消耗模板位置（包括占位符）
        if idx_template < len(template_num) and (templ_char != '-' or template_num[idx_template].startswith('-')):
            num = template_num[idx_template]
            # 占位符编号如 '-1'
            if num.startswith('-'):
                print(f"  -> placeholder: 编号 {num}, 填 '-' ")
                aligned_nums.append(num)
                aligned_seqs.append('-')
                last_num = num
                idx_template += 1
                ins_count = 1
                continue
            # 正常模板位置
            if templ_char != '-':
                print(f"  -> match: 编号 {num}, 碱基 '{targ_char}'")
                aligned_seqs.append(targ_char)
            else:
                print(f"  -> deletion: 编号 {num}, 填 '-' ")
                aligned_seqs.append('-')
            aligned_nums.append(num)
            last_num = num
            idx_template += 1
            ins_count = 1
        # 插入事件
        elif templ_char == '-' and targ_char != '-':
            base = last_num or template_num[0]
            num = f"{base}i{ins_count}"
            print(f"  -> insertion: 基底 '{base}', 编号 {num}, 碱基 '{targ_char}'")
            aligned_nums.append(num)
            aligned_seqs.append(targ_char)
            ins_count += 1
        else:
            print("  -> both gap, skip")

    # 4) 填充剩余模板编号
    while idx_template < len(template_num):
        num = template_num[idx_template]
        print(f"[Debug] 填充尾部模板编号 {num} with '-' ")
        aligned_nums.append(num)
        aligned_seqs.append('-')
        idx_template += 1

    # 5) 校验
    if validate_alignment(target_seq, aligned_seqs):
        print("[Info] 全碱基校验通过")
    else:
        print("[Warning] 对齐碱基不匹配目标序列")

    print(f"[Result] aligned_numbering = {aligned_nums}")
    print(f"[Result] aligned_sequence  = {aligned_seqs}")
    return {"aligned_numbering": aligned_nums, "aligned_sequence": aligned_seqs}


if __name__ == '__main__':
    tpl_nums = ['-1','1a','2b','3','4','5','6','7','8','9','10']
    tpl_seq  = '-AUGCUAAAAA'
    tgt_seq  = 'AUGGGCUUAAAAA'
    print('=== 测试 1 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)
    tpl_nums = ['1','2','3','4','5']
    tpl_seq  = 'AUGCU'
    tgt_seq  = 'AUGGCU'
    print('=== 测试 2 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)
    print('=== 测试结束 ===')
    # 测试示例 1
    tpl_nums = ['-1','1','2','3','4','5','6','7','8','9','10']
    tpl_seq  = '-AUGCUAAAAA'
    tgt_seq  = 'AUGGCU'
    print('=== 测试 1 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)

    # 测试示例 2
    tpl_nums = ['1','2','3','4','5']
    tpl_seq  = 'AUGCU'
    tgt_seq  = 'AUGGCU'
    print('=== 测试 2 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)
    print('=== 测试结束 ===')
    # 测试
    tpl_nums = ['-1','1','2','3','4','5','6','7','8','9','10']
    tpl_seq  = '-AUGCUAAAAA'
    tgt_seq  = 'AUGGCU'
    print('=== DP 对齐测试 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)
    print('=== 测试结束 ===')
    # 测试
    tpl_nums = ['1','2','3','4','5']
    tpl_seq  = 'AUGCU'
    tgt_seq  = 'AUGGCU'
    print('=== DP 对齐测试 ===')
    _align_with_dp(tpl_nums, tpl_seq, tgt_seq)
    print('=== 测试结束 ===')
