# -*- coding: utf-8 -*-  # UTF-8
"""
对齐包装层：保持外部调用签名不变，内部转到你原来的实现
"""  # 模块说明

from typing import Tuple  # 类型注解
from pythonscript.align import perform_full_alignment as _run_alignment  # 引入你原先实现

def perform_full_alignment(target_seq: str, output_csv_path: str, anticode: str = "", use_llm: bool = True) -> Tuple[str, str]:
    """
    完整对齐流水线（外部统一入口）：
    参数保持不变：序列 / 输出路径 / 反密码子 / 是否优先LLM
    返回：模板名、两行CSV原文
    """  # 函数说明
    return _run_alignment(  # 直接调用你已有逻辑
        target_seq=target_seq,  # 目标序列
        output_csv_path=output_csv_path,  # 输出路径（此处多用 /dev/null）
        anticode=anticode,  # 反密码子（可选）
        use_llm=use_llm,  # 是否优先LLM
    )  # 返回(模板名, csv原文)
