# -*- coding: utf-8 -*-  # UTF-8
"""
区域常量与参数校验：提供白名单、去重/规范化、冲突检测
"""  # 模块说明

from typing import Any, List, Tuple  # 类型注解

# 与你前端/模型一致的区域白名单
REGION_OPTIONS: List[str] = [
    "AA_Arm_5prime",  # 5' 氨基酸臂
    "D_Loop",         # D 环
    "Anticodon_Arm",  # 反密码子臂
    "Variable_Loop",  # 变量环
    "T_Arm",          # T 臂
    "AA_Arm_3prime",  # 3' 氨基酸臂
]

def _normalize_region_list(v: Any) -> List[str]:
    """将区域参数统一为去重后的字符串列表"""  # 函数说明
    if not v:  # 空值直接返回空列表
        return []  # 空列表
    if isinstance(v, str):  # 单个字符串转列表
        v = [v]  # 包装成列表
    out: List[str] = []  # 输出容器
    seen = set()  # 去重集合
    for x in v:  # 遍历输入
        if not isinstance(x, str):  # 忽略非字符串
            continue  # 跳过非法
        name = x.strip()  # 去掉两端空白
        if name and name not in seen:  # 非空且未出现
            out.append(name)  # 加入结果
            seen.add(name)  # 做去重标记
    return out  # 返回结果

def validate_and_prepare_regions(prefer_regions: Any, freeze_regions: Any) -> Tuple[List[str], List[str]]:
    """
    校验并规范 prefer/freeze：
    - 名称必须在白名单
    - freeze 不能全选
    - 两者不能有交集
    - prefer 为空则回退默认 ["Variable_Loop"]
    """  # 函数说明
    pref = _normalize_region_list(prefer_regions)  # 规范化 prefer
    frz = _normalize_region_list(freeze_regions)  # 规范化 freeze

    invalid_pref = [x for x in pref if x not in REGION_OPTIONS]  # prefer 非法项
    invalid_frz = [x for x in frz if x not in REGION_OPTIONS]  # freeze 非法项
    if invalid_pref or invalid_frz:  # 若任一存在非法
        msg = []  # 拼接错误信息
        if invalid_pref:  # prefer 非法描述
            msg.append(f"Unknown prefer_regions: {invalid_pref}")  # 加入消息
        if invalid_frz:  # freeze 非法描述
            msg.append(f"Unknown freeze_regions: {invalid_frz}")  # 加入消息
        raise ValueError("; ".join(msg))  # 抛出校验异常

    if len(frz) == len(REGION_OPTIONS):  # freeze 不允许覆盖全部区域
        raise ValueError("freeze_regions cannot include all regions.")  # 抛异常

    overlap = sorted(set(pref) & set(frz))  # 交集检测
    if overlap:  # 如有交集
        raise ValueError(f"freeze_regions and prefer_regions cannot overlap: {overlap}")  # 抛异常

    if not pref:  # prefer 为空时使用默认
        pref = ["Variable_Loop"]  # 默认偏好 Variable_Loop

    return pref, frz  # 返回校验后的两个列表
