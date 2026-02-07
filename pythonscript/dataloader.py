#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块: compare_input_template
功能: 给定一条 RNA 序列，查找最匹配的 tRNA 模板，并提取输入序列和模板在六个结构区的对应片段。
可被其他脚本 import 调用。
"""
import os
import re
from typing import Dict, Any
from pythonscript.find_best_match import find_best_match
from pythonscript.group_by_region import group_by_region
from pythonscript.tRNAscan_SE_annotator import annotate_trna_sequences

# 六个区域名称列表，顺序重要
REGIONS = [
    "Aminoacyl arm 5' end",
    "D loop + D stem",
    "Anticodon loop + Anticodon stem",
    "Variable loop",
    "T loop + T stem",
    "Aminoacyl arm 3' end"
]

def _norm_u(seq: str) -> str:
    return (seq or "").replace("T", "U").replace("t", "U").upper()

def _patch_tail_cca(full_input_seq: str, last_region_seq: str) -> str:
    """
    若原始输入序列在最后一段之后还有 1~3 个尾端碱基，并且形如 A/CA/CCA，
    则把这些碱基并入最后一段。避免 74–76 丢失。
    """
    s_full = _norm_u(full_input_seq)
    s_last = _norm_u(last_region_seq)
    if not s_full or not s_last:
        return last_region_seq

    # 尝试在 full 中寻找最后一段的最后一次出现位置
    pos = s_full.rfind(s_last)
    if pos < 0:
        return last_region_seq

    extra = s_full[pos + len(s_last):]  # 最后一段之后的“尾巴”
    # 只看末尾最多 3 个碱基，并且仅保留 A/C/G/U
    extra = re.sub(r"[^ACGU]", "", extra)[:3]

    # 典型尾巴：A / CA / CCA
    if extra in ("A", "CA", "CCA"):
        return last_region_seq + extra
    return last_region_seq

def compare_input_template(
    input_seq: str,
    tsv_path: str = os.path.join('data', 'tRNAs_flat.tsv')
) -> Dict[str, Any]:
    """
    输入: 一条 RNA 序列。
    步骤：
      1) find_best_match -> 模板
      2) group_by_region -> 模板各区编号/序列
      3) annotate_trna_sequences -> 输入序列分区
      4) ★ 修补 3′ 端 CCA：把最后一段的尾巴并入（若存在 A/CA/CCA）
    返回:
    {
      'template_name': str,
      'template_flat_seq': str,
      'regions': {
         region_name: {
           'input_seq': str,
           'template_numbering': List[str],
           'template_seq': str
         }, ...
      }
    }
    """
    raw_input = _norm_u(input_seq)

    # 1) 找模板
    best = find_best_match(raw_input, tsv_path)
    if not best:
        return {}
    template_name = best['seqname']
    template_flat_seq = best['flat_seq']

    # 2) 模板分区
    template_regions = group_by_region(best)

    # 3) 输入序列注释
    annotated = annotate_trna_sequences({'input': raw_input})
    if not annotated:
        return {}
    input_annot = annotated[0]  # 单条

    # 4) 组装 & 修补尾巴
    result: Dict[str, Any] = {
        'template_name': template_name,
        'template_flat_seq': template_flat_seq,
        'regions': {}
    }

    for region in REGIONS:
        tpl = template_regions.get(region, {})
        inp_seq = _norm_u(input_annot.get(region, ""))

        if region == "Aminoacyl arm 3' end":
            # ★ 把 3′ 端尾巴（A/CA/CCA）并到最后一段
            inp_seq_patched = _patch_tail_cca(raw_input, inp_seq)

            # 可选：若模板编号包含 74/75/76，给个提示方便排查
            tnums = tpl.get('numbering', []) or []
            if any(n in tnums for n in ("74", "75", "76")) and inp_seq_patched != inp_seq:
                print(f"[Patch] 3' tail appended: '{inp_seq}' -> '{inp_seq_patched}'")

            inp_seq = inp_seq_patched

        result['regions'][region] = {
            'input_seq': inp_seq,
            'template_numbering': tpl.get('numbering', []),
            'template_seq': tpl.get('sequence', '')
        }

    return result

if __name__ == '__main__':
    # 示例测试
    test_seq = "AAAAAAUUAGUUUAAUCAAAAACCUUAGUAUGUCAAACUAAAAAAAUUAGAUCAUCUAAUAUUUUUUACCA"
    info = compare_input_template(test_seq)
    if not info:
        print("No match or annotation failed.")
    else:
        print(f"Template: {info['template_name']}")
        print(f"Template flat seq: {info['template_flat_seq']}\n")
        for region, data in info['regions'].items():
            print(f"{region}:")
            print(f"  Input seq           : {data['input_seq']}")
            print(f"  Template numbering  : {data['template_numbering']}")
            print(f"  Template seq        : {data['template_seq']}\n")
