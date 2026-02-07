# -*- coding: utf-8 -*-
# ==============================================
# File: pythonscript/llm_postcheck.py
# ----------------------------------------------
# Dispatcher for 6 standard regions.
# Each region lives in its own file under pythonscript/postcheck/.
# Variable loop has a full implementation; others are placeholders for now.
# ==============================================

from __future__ import annotations
from typing import List, Dict, Any

# 标准 6 区段
regions = [
    "Aminoacyl arm 5' end",
    "D loop + D stem",
    "Anticodon loop + Anticodon stem",
    "Variable loop",
    "T loop + T stem",
    "Aminoacyl arm 3' end",
]

# 每个区段都有独立文件与入口
from .postcheck import (
    postcheck_aminoacyl_5,
    postcheck_d_region,
    postcheck_anticodon,
    postcheck_variable_loop,
    postcheck_t_region,
    postcheck_aminoacyl_3,
)

_REGION_HANDLERS = {
    "Aminoacyl arm 5' end":            postcheck_aminoacyl_5,
    "D loop + D stem":                 postcheck_d_region,
    "Anticodon loop + Anticodon stem": postcheck_anticodon,
    "Variable loop":                   postcheck_variable_loop,
    "T loop + T stem":                 postcheck_t_region,
    "Aminoacyl arm 3' end":            postcheck_aminoacyl_3,
}

def postcheck_alignment(blocks: List[Dict[str, Any]], use_llm: bool = True) -> List[Dict[str, Any]]:
    """
    对每个 block 按 region 分发到对应后校对函数。
    - use_llm=True（默认）：保持原有行为，分发到各区段后校对（其中可能调用大模型）。
    - use_llm=False：完全跳过后校对（包括一切大模型相关步骤），直接原样返回 blocks。

    说明：
    之所以在此做“硬旁路（bypass）”，是因为各区段实现的函数签名目前只接收 block，
    且是否启用 LLM 的开关通常在各文件内部用环境变量常量化读取，运行期很难
    通过参数动态传递进去并生效。为了保证行为可预测，这里选择在调度层面直接跳过。
    """
    if not use_llm:
        # 直接返回：不进入任何区段的 postcheck，等价于“不要大模型”
        return list(blocks)

    out: List[Dict[str, Any]] = []
    for blk in blocks:
        region = (blk or {}).get("region", "")
        handler = _REGION_HANDLERS.get(region)
        if handler is None:
            out.append(blk)
            continue
        try:
            out.append(handler(blk))
        except Exception as e:
            print(f"[POSTCHECK][ERROR] handler for region '{region}' raised: {repr(e)}. Passing through.")
            out.append(blk)
    return out
