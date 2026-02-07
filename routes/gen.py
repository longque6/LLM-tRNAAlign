# -*- coding: utf-8 -*-
"""
/trnagen 路由（生成序列）——移除环境变量依赖，改为从 config/gen_models.py 的注册表按名称加载。
"""

import time  # 计时
import threading  # 并发锁，防止首次加载冲突
from typing import Any, Dict, Tuple  # 类型注解

from flask_restx import Namespace, Resource, fields  # RESTX 组件
from flask import request  # 获取请求体
import torch  # 用于选择/校验设备

# 导入“写死”的模型注册表
from config.gen_models import GEN_MODEL_REGISTRY  # 固定的模型清单

# 导入你已有的生成实现（保持不变）
from pythonscript.gen.gen_mvp import (
    load_embedding_and_model,  # 加载 embedding 与模型
    generate_one_mvp_regions,  # 单条生成
    generate_batch_mvp_regions,  # 批量生成
)

# =========================
# 线程安全的模型缓存（多模型）
# =========================
_MODEL_LOCK = threading.Lock()  # 防止并发首次加载相同模型
# 缓存结构：{ model_name: (embedding, model, device_str) }
_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}

def _pick_torch_device(device_str: str) -> torch.device:
    """把 'cuda'/'cuda:1'/'cpu' 转成 torch.device，并在不可用时降级到 cpu。"""
    ds = (device_str or "cuda").strip().lower()  # 规范字符串
    if ds.startswith("cuda"):
        if torch.cuda.is_available():  # CUDA 可用则按用户指定
            try:
                return torch.device(ds)  # 可能是 'cuda' 或 'cuda:0'
            except Exception:
                return torch.device("cuda")
        else:
            return torch.device("cpu")  # 无 CUDA 时降级
    return torch.device("cpu")  # 默认 CPU

def _ensure_model_loaded(model_name: str) -> Tuple[Any, Any, torch.device]:
    """
    按名称加载（或返回缓存的）模型；名称必须存在于 GEN_MODEL_REGISTRY。
    返回：(embedding, model, torch_device)
    """
    if model_name not in GEN_MODEL_REGISTRY:  # 名称合法性检查
        raise ValueError(f"Unknown model_name: {model_name}")

    with _MODEL_LOCK:  # 并发保护
        if model_name in _MODEL_CACHE:  # 已缓存则直接返回
            em, mdl, dev_str = _MODEL_CACHE[model_name]
            return em, mdl, _pick_torch_device(dev_str)

        # 首次加载
        cfg = GEN_MODEL_REGISTRY[model_name]  # 取出写死的 ckpt 与 device
        ckpt_path = cfg["ckpt"]  # 权重路径
        device_str = cfg.get("device", "cuda")  # 设备字符串
        device = _pick_torch_device(device_str)  # 转换为 torch.device

        t0 = time.time()  # 计时
        print(f"[trnagen] loading model '{model_name}' (ckpt={ckpt_path}, device={device})")  # 日志
        embedding, model = load_embedding_and_model(ckpt_path, device=device)  # 实际加载
        _MODEL_CACHE[model_name] = (embedding, model, device_str)  # 放入缓存
        print(f"[trnagen] model '{model_name}' ready in {time.time()-t0:.2f}s")  # 耗时日志
        return embedding, model, device  # 返回句柄

# =========================
# RESTX 命名空间与模型定义
# =========================
def register_gen_namespace(api) -> None:
    """对外注册 /trnagen 命名空间与路由。"""
    ns_gen = Namespace("trnagen", description="Sequence generation using local Transformer")  # 命名空间

    # 请求/响应体（单条）
    gen_one_req = ns_gen.model("GenOneRequest", {
        # 新增 model_name，默认 "default"
        "model_name": fields.String(default="default", description="Choose a model from server registry"),
        "seed_seq": fields.String(required=True, description="Seed tRNA sequence"),
        "rounds": fields.Integer(default=3),
        "mask_frac": fields.Float(default=0.20),
        "temperature": fields.Float(default=0.9),
        "top_p": fields.Float(default=0.9),
        "top_k": fields.Integer(default=0),
        "mask_k": fields.Integer(default=5),
        "prefer_regions": fields.List(fields.String, default=["Variable_Loop"]),
        # 以下保持向后兼容
        "prefer_positions": fields.List(fields.Integer, default=[]),
        "freeze_positions": fields.List(fields.Integer, default=[]),
        "freeze_regions": fields.List(fields.String, default=[]),
        "force_positions": fields.Raw(default={}),
        "prefer_capacity_ratio": fields.Float(default=0.6),
        "verbose": fields.Boolean(default=False),
    })
    gen_one_res = ns_gen.model("GenOneResponse", {
        "sequence": fields.String(description="Generated sequence"),
        "elapsed_sec": fields.Float(description="Latency in seconds"),
        "model_name": fields.String(description="Model actually used")
    })

    # 请求/响应体（批量）
    gen_batch_req = ns_gen.model("GenBatchRequest", {
        "model_name": fields.String(default="default", description="Choose a model from server registry"),
        "seed_seq": fields.String(required=True, description="Seed sequence"),
        "num_samples": fields.Integer(default=3),
        "rounds": fields.Integer(default=3),
        "mask_frac": fields.Float(default=0.20),
        "temperature": fields.Float(default=0.95),
        "top_p": fields.Float(default=0.9),
        "top_k": fields.Integer(default=0),
        "mask_k": fields.Integer(default=5),
        "prefer_regions": fields.List(fields.String, default=["Variable_Loop"]),
        "min_hd": fields.Integer(default=2),
        "oversample_factor": fields.Integer(default=1),
        "rerank_min_hd": fields.Integer(default=6),
        "gc_low": fields.Float(default=0.42),
        "gc_high": fields.Float(default=0.66),
        "ensure_reach": fields.Boolean(default=True),
        "max_attempts": fields.Integer(default=6),
        # 兼容可选字段
        "prefer_positions": fields.List(fields.Integer, default=[]),
        "freeze_positions": fields.List(fields.Integer, default=[]),
        "freeze_regions": fields.List(fields.String, default=[]),
        "prefer_capacity_ratio": fields.Float(default=0.6),
        "rerank_top_k": fields.Integer(default=None),
        "dedup": fields.Boolean(default=True),
        "min_region_lens": fields.Raw(default={
            'AA_Arm_5prime': 5,
            'D_Loop': 8,
            'Anticodon_Arm': 8,
            'Variable_Loop': 0,
            'T_Arm': 8,
            'AA_Arm_3prime': 5,
        }),
        "verbose": fields.Boolean(default=False),
    })
    gen_batch_res = ns_gen.model("GenBatchResponse", {
        "sequences": fields.List(fields.String, description="Generated sequences"),
        "count": fields.Integer,
        "elapsed_sec": fields.Float,
        "model_name": fields.String(description="Model actually used")
    })

    # =========================
    #   路由1：/trnagen/one
    # =========================
    @ns_gen.route("/one")
    class GenOne(Resource):  # 单条生成资源
        @ns_gen.expect(gen_one_req, validate=True)  # 声明请求体
        @ns_gen.response(200, "Success", gen_one_res)  # 声明响应体
        def post(self):
            """Single generation of sequences using a specified model"""
            data = request.get_json(force=True) or {}  # 取 JSON
            model_name = str(data.get("model_name", "default"))  # 取模型名，无则 default
            seed_seq = (data.get("seed_seq") or "").strip()  # 取种子序列
            if not seed_seq:  # 参数校验
                ns_gen.abort(400, "Field `seed_seq` is required.")  # 返回 400
                return

            # 确保模型可用（会自动缓存）
            embedding, model, device = _ensure_model_loaded(model_name)  # 加载/取缓存

            # 组装可选参数（保持你原始接口习惯）
            rounds = int(data.get("rounds", 3))
            mask_frac = float(data.get("mask_frac", 0.20))
            temperature = float(data.get("temperature", 0.9))
            top_p = float(data.get("top_p", 0.9))
            top_k = int(data.get("top_k", 0))
            mask_k = data.get("mask_k", 5)
            prefer_regions = data.get("prefer_regions", ["Variable_Loop"])
            prefer_positions = data.get("prefer_positions", [])
            freeze_positions = data.get("freeze_positions", [])
            freeze_regions = data.get("freeze_regions", [])
            force_positions = data.get("force_positions", {})
            prefer_capacity_ratio = float(data.get("prefer_capacity_ratio", 0.6))
            verbose = bool(data.get("verbose", False))

            t0 = time.time()  # 计时
            try:
                seq = generate_one_mvp_regions(
                    seed_seq,
                    embedding,
                    model,
                    device,
                    rounds=rounds,
                    mask_frac=mask_frac,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    freeze_positions=freeze_positions,
                    force_positions=force_positions,
                    mask_k=mask_k,
                    freeze_regions=freeze_regions,
                    prefer_positions=prefer_positions,
                    prefer_regions=prefer_regions,
                    prefer_capacity_ratio=prefer_capacity_ratio,
                    verbose=verbose,
                )
            except Exception as e:
                ns_gen.abort(500, f"Generation failed: {e}")  # 异常转 500
                return

            return {
                "sequence": seq,  # 生成的序列
                "elapsed_sec": round(time.time() - t0, 3),  # 耗时
                "model_name": model_name,  # 实际使用的模型名
            }, 200

    # =========================
    #   路由2：/trnagen/batch
    # =========================
    @ns_gen.route("/batch")
    class GenBatch(Resource):  # 批量生成资源
        @ns_gen.expect(gen_batch_req, validate=True)  # 声明请求体
        @ns_gen.response(200, "Success", gen_batch_res)  # 声明响应体
        def post(self):
            """Batch generation of sequences using a specified model"""
            data = request.get_json(force=True) or {}  # 取 JSON
            model_name = str(data.get("model_name", "default"))  # 模型名
            seed_seq = (data.get("seed_seq") or "").strip()  # 种子序列
            if not seed_seq:
                ns_gen.abort(400, "Field `seed_seq` is required.")  # 无则报错
                return

            embedding, model, device = _ensure_model_loaded(model_name)  # 取模型句柄

            # 保持与 generate_batch_mvp_regions 一致的参数对齐
            kwargs = dict(
                num_samples=int(data.get("num_samples", 3)),
                rounds=int(data.get("rounds", 3)),
                mask_frac=float(data.get("mask_frac", 0.20)),
                temperature=float(data.get("temperature", 0.95)),
                top_p=float(data.get("top_p", 0.9)),
                top_k=int(data.get("top_k", 0)),
                freeze_positions=data.get("freeze_positions", []),
                force_positions=data.get("force_positions", {}),
                min_hd=int(data.get("min_hd", 2)),
                dedup=bool(data.get("dedup", True)),
                mask_k=data.get("mask_k", 5),
                freeze_regions=data.get("freeze_regions", []),
                prefer_positions=data.get("prefer_positions", []),
                prefer_regions=data.get("prefer_regions", ["Variable_Loop"]),
                prefer_capacity_ratio=float(data.get("prefer_capacity_ratio", 0.6)),
                oversample_factor=int(data.get("oversample_factor", 1)),
                rerank_min_hd=int(data.get("rerank_min_hd", 6)),
                rerank_top_k=data.get("rerank_top_k", None),
                gc_low=float(data.get("gc_low", 0.42)),
                gc_high=float(data.get("gc_high", 0.66)),
                min_region_lens=data.get("min_region_lens", {
                    'AA_Arm_5prime': 5,
                    'D_Loop': 8,
                    'Anticodon_Arm': 8,
                    'Variable_Loop': 0,
                    'T_Arm': 8,
                    'AA_Arm_3prime': 5,
                }),
                ensure_reach=bool(data.get("ensure_reach", True)),
                max_attempts=int(data.get("max_attempts", 6)),
                verbose=bool(data.get("verbose", True)),
            )

            t0 = time.time()  # 计时
            try:
                seqs = generate_batch_mvp_regions(  # 批量生成调用
                    seed_seq,
                    embedding,
                    model,
                    device,
                    **kwargs
                )
            except Exception as e:
                ns_gen.abort(500, f"Batch generation failed: {e}")  # 异常转 500
                return

            return {
                "sequences": seqs,  # 生成的序列数组
                "count": len(seqs),  # 数量
                "elapsed_sec": round(time.time() - t0, 3),  # 耗时
                "model_name": model_name,  # 实际使用的模型名
            }, 200

    # 最后把命名空间挂到 API 上
    api.add_namespace(ns_gen)  # 注册 /trnagen
