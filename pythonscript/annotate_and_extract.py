#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给定一条 RNA 序列，找到最匹配的 tRNA 模板，并提取其二级结构和六段区域的编号及序列。
可被其他脚本调用。
"""
import os
import csv
from typing import Dict, Any
from pythonscript.find_best_match import find_best_match
from pythonscript.number_to_region import number_to_region

# 预定义的 tRNA 六个区域名称
REGIONS = [
    "Aminoacyl arm 5' end",
    "D loop + D stem",
    "Anticodon loop + Anticodon stem",
    "Variable loop",
    "T loop + T stem",
    "Aminoacyl arm 3' end"
]


def annotate_and_extract(
    input_seq: str,
    tsv_path: str = os.path.join('data', 'tRNAs_flat.tsv'),
) -> Dict[str, Any]:
    """
    对 input_seq 进行比对，找到最匹配的 tRNA 模板。
    返回结构：
    {
      'template_name': str,
      'secondary_structure': str,
      'regions': {
          region: {
              'numbering': List[str],
              'sequence': List[str]
          }, ...
      }
    }
    """
    # 找到最匹配的记录
    best = find_best_match(input_seq, tsv_path)
    if best is None:
        return {}

    template_name = best['seqname']
    # 假设 'SecondaryStructure' 字段存在于 best，否则空字符串
    sec_struct = best.get('SecondaryStructure', '')

    # 过滤出所有可编号的列（排除元数据）
    all_cols = [c for c in best.keys()
                if c not in ('seqname', 'score', 'flat_seq', 'alignment', 'SecondaryStructure')]

    regions_data = {}
    for region in REGIONS:
        numbers = []
        seqs = []
        for col in all_cols:
            try:
                if number_to_region(col) == region:
                    numbers.append(col)
                    seqs.append(best[col])
            except ValueError:
                # 非编号列或不在任何区域，忽略
                continue
        regions_data[region] = {
            'numbering': numbers,
            'sequence': seqs
        }

    return {
        'template_name': template_name,
        'secondary_structure': sec_struct,
        'regions': regions_data
    }


if __name__ == '__main__':
    # 简单测试
    test_seq = (
        "CCCCCCCCGGUGGUGCAGUGGUUAAGGCGCCGCCUUUAACGGCGGAGGCCCGGGUUCGAUUCCCGGUCGGGGAC"
    )
    info = annotate_and_extract(test_seq)
    if not info:
        print("No matching template found.")
    else:
        print(f"Template: {info['template_name']}")
        print(f"Secondary structure: {info['secondary_structure']}\n")
        for region, data in info['regions'].items():
            print(f"{region}:")
            print(f"  Numbering: {data['numbering']}")
            print(f"  Sequence : {data['sequence']}\n")
