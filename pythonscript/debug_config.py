#!/usr/bin/env python3  # 解释器路径，保证脚本可直接执行
# -*- coding: utf-8 -*-  # 文件编码，允许中文注释
# 统一 debug 配置模块，集中管理 TRNAALIGN_* 环境变量开关  # 模块用途说明
# 所有 debug 开关只在这里读取，其他模块只 import 这个文件  # 设计原则说明
from __future__ import annotations  # 允许前向类型注解（Python 3.7+）
import os  # 读取环境变量用
# 读取环境变量并解析为布尔值  # 函数用途说明
def env_flag(name: str, default: bool = False) -> bool:  # name=变量名，default=默认值
    val = os.getenv(name)  # 从环境变量读取字符串
    if val is None:  # 未设置则走默认值
        return default  # 返回默认布尔值
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")  # 常见“关闭”取值视为 False
# 读取环境变量并解析为整数  # 函数用途说明
def env_int(name: str, default: int) -> int:  # name=变量名，default=默认值
    try:  # 捕获非法整数
        return int(os.getenv(name, str(default)))  # 有值就转为 int，没有就用默认值
    except Exception:  # 转换失败时兜底
        return default  # 返回默认值
# 对齐相关 debug  # 功能分组说明
DEBUG_ALIGN = env_flag("TRNAALIGN_DEBUG_ALIGN", True)  # 对齐主 debug 总开关
DEBUG_ALIGN_FULL = env_flag("TRNAALIGN_DEBUG_ALIGN_FULL", True)  # 是否输出完整对齐细节
DEBUG_ALIGN_MAXLEN = max(40, env_int("TRNAALIGN_DEBUG_MAXLEN", 200))  # 对齐日志截断长度
# 模板检索相关 debug  # 功能分组说明
DEBUG_MATCH = env_flag("TRNAALIGN_DEBUG_MATCH", False)  # 模板检索与粗筛 debug 开关
# 注释/分段相关 debug  # 功能分组说明
DEBUG_ANNOTATE = env_flag("TRNAALIGN_DEBUG_ANNOTATE", True)  # 分段注释 debug 总开关
DEBUG_ANNOTATE_FULL = env_flag("TRNAALIGN_DEBUG_ANNOTATE_FULL", True)  # 是否输出完整结构内容
DEBUG_ANNOTATE_SUMMARY = env_flag("TRNAALIGN_DEBUG_ANNOTATE_SUMMARY", True)  # 是否输出每条序列汇总
