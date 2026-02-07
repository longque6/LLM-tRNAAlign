# -*- coding: utf-8 -*-  # UTF-8编码
"""
生成模型服务：设备选择、懒加载（线程安全）、全局缓存
"""  # 模块用途

import os  # 读取环境变量
import time  # 简单计时打点
import threading  # 线程锁保证首次加载并发安全
from typing import Tuple  # 类型注解

import torch  # PyTorch
from pythonscript.gen.gen_mvp import load_embedding_and_model  # 你现有的生成模型加载函数

# 全局缓存变量（仅在本模块内维护）
_GEN_LOCK = threading.Lock()  # 保护首次加载
_GEN_READY = False  # 是否已加载
_GEN_DEVICE = None  # torch.device 缓存
_GEN_EMBEDDING = None  # embedding 实例
_GEN_MODEL = None  # 生成模型实例
_GEN_CKPT_PATH = None  # 权重路径缓存

def pick_device(env_val: str | None) -> torch.device:
    """根据环境变量选择设备，不合法则自动回退"""  # 函数说明
    if env_val:  # 如用户显式指定
        v = env_val.strip().lower()  # 统一小写
        if v == "cuda" and torch.cuda.is_available():  # CUDA可用则选
            return torch.device("cuda")  # 返回CUDA设备
        if v == "cpu":  # 允许强制CPU
            return torch.device("cpu")  # 返回CPU
        print(f"[WARN] GEN_DEVICE={env_val} 不可用，自动检测中…")  # 警告提示
    # 自动检测，有CUDA则用CUDA，否则CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 返回自动设备

def ensure_gen_model_loaded() -> Tuple[torch.nn.Module, torch.nn.Module, torch.device]:
    """懒加载生成模型（线程安全），返回(embedding, model, device)"""  # 函数说明
    global _GEN_READY, _GEN_DEVICE, _GEN_EMBEDDING, _GEN_MODEL, _GEN_CKPT_PATH  # 使用全局缓存
    if _GEN_READY:  # 如果已经加载则直接返回
        return _GEN_EMBEDDING, _GEN_MODEL, _GEN_DEVICE  # 返回缓存对象
    with _GEN_LOCK:  # 加锁防止并发重复加载
        if _GEN_READY:  # 双重检查，锁内再判一次
            return _GEN_EMBEDDING, _GEN_MODEL, _GEN_DEVICE  # 已加载直接返回
        _GEN_CKPT_PATH = os.environ.get("GEN_CKPT", "./checkpoint.pth")  # 读取权重路径（默认 ./checkpoint.pth）
        _GEN_DEVICE = pick_device(os.environ.get("GEN_DEVICE"))  # 确定设备（支持覆盖）
        t0 = time.time()  # 记录起始时间
        print(f"[INFO] Loading gen model: ckpt={_GEN_CKPT_PATH}, device={_GEN_DEVICE}")  # 打印加载信息
        embedding, model = load_embedding_and_model(_GEN_CKPT_PATH, device=_GEN_DEVICE)  # 调用你现有函数加载
        _GEN_EMBEDDING = embedding  # 缓存embedding
        _GEN_MODEL = model  # 缓存model
        _GEN_READY = True  # 标记加载完成
        print(f"[INFO] Gen model ready in {time.time()-t0:.2f}s")  # 打印耗时
        return _GEN_EMBEDDING, _GEN_MODEL, _GEN_DEVICE  # 返回对象三件套
