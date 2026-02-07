# -*- coding: utf-8 -*-  # 声明文件编码为UTF-8，支持中文注释
"""
/readthrough 命名空间
- 提供基于本地通读效率模型的预测接口
- 使用 pythonscript/readthrough/inference.py 中的 predict_suptrna_efficiency
- 模型路径与设备从 config/gen_models.py 的 READ_MODEL_REGISTRY 获取
"""

from typing import Tuple  # 类型注解（可选）
from flask_restx import Namespace, Resource, fields  # 引入RESTX命名空间、资源、字段定义
from flask import request  # 获取请求体
import os  # 用于路径处理或将来扩展
import json  # 用于调试时安全打印
from config.gen_models import READ_MODEL_REGISTRY  # 读取通读模型注册表
from pythonscript.readthrough.inference import predict_suptrna_efficiency  # 核心推理函数
from pythonscript.readthrough.inference import CKPT_PATH as DEFAULT_READTHROUGH_CKPT  # 默认权重路径后备

# 工厂函数，向传入的 Api 实例注册本命名空间
def register_read_namespace(api) -> None:
    # 创建一个命名空间，路径前缀为 /readthrough
    ns = Namespace("readthrough", description="Readthrough efficiency inference APIs")

    # Swagger请求模型定义（POST /predict 的请求体）
    predict_req = ns.model("ReadthroughPredictRequest", {
        "sequence": fields.String(  # 必填，待预测的 tRNA 序列
            required=True,
            description="Sup-tRNA sequence with characters A/U/G/C.",
            example="GGUCUCGUGGCCCAAUGGUUAAGGCGCUUGACUACGGAUCAAGAGAUUCCAGGUUCGACUCCUGGCGGGAUCG"
        ),
        "mc_samples": fields.Integer(  # 可选，MC Dropout 采样次数
            required=False,
            default=50,
            description="Number of MC-Dropout samples (higher → wider CI)."
        ),
        "ckpt_name": fields.String(  # 可选，读取注册表中某个模型名；默认用 'readthrough'
            required=False,
            default="readthrough",
            description="Model name in READ_MODEL_REGISTRY (default: 'readthrough')."
        )
    })

    # Swagger响应模型定义（简化展示；真实返回体包含更多字段）
    predict_res = ns.model("ReadthroughPredictResponse", {
        "pred_raw": fields.Float(description="Predicted readthrough on raw scale"),
        "pred_log": fields.Float(description="Predicted value in log-standardized space"),
        "std_raw": fields.Float(description="Std of raw predictions from MC-Dropout"),
        "ci95_raw": fields.List(fields.Float, description="95% CI on raw scale [low, high]"),
        "confidence_pct": fields.Float(description="Heuristic confidence in %"),
        "structure": fields.String(description="Secondary structure string"),
        "region_seqs": fields.Raw(description="Six-region sequences used by the model"),
        "ckpt_used": fields.String(description="Which checkpoint was actually used")
    })

    # 健康检查接口（GET /health）
    @ns.route("/health")
    class Health(Resource):
        @ns.response(200, "OK")
        def get(self):
            # 简单返回固定结果，方便K8s或外部健康检查
            return {"ok": True, "service": "readthrough"}, 200

    # 主推理接口（POST /predict）
    @ns.route("/predict")
    class Predict(Resource):
        @ns.expect(predict_req, validate=True)  # 启用请求体验证
        @ns.response(200, "Success", predict_res)  # 成功响应示意
        @ns.response(400, "Bad Request")  # 参数错误
        @ns.response(500, "Internal Server Error")  # 内部错误
        def post(self):
            # 解析JSON请求体
            data = request.get_json(force=True) or {}
            seq = (data.get("sequence") or "").strip()  # 读取序列并去除空白
            if not seq:  # 校验序列不能为空
                ns.abort(400, "Field `sequence` is required and must be non-empty.")
                return  # 类型检查完善

            # 读取可选参数 mc_samples
            mc_samples = int(data.get("mc_samples", 50))

            # 决定使用的模型名（从注册表中检索路径与设备）
            model_name = (data.get("ckpt_name") or "readthrough").strip()

            # 从 READ_MODEL_REGISTRY 中获取模型配置；若不存在则回退到 inference 内置默认
            if model_name in READ_MODEL_REGISTRY:
                ckpt_path = READ_MODEL_REGISTRY[model_name].get("ckpt") or DEFAULT_READTHROUGH_CKPT
            else:
                ckpt_path = DEFAULT_READTHROUGH_CKPT

            # 执行推理（readthrough 的 inference 已经内部固定为CPU，符合注册表配置）
            try:
                out = predict_suptrna_efficiency(
                    seq=seq,                  # 传入待预测序列
                    mc_samples=mc_samples,   # 传入MC次数字段
                    ckpt_path=ckpt_path      # 传入选定的权重路径
                )
            except Exception as e:
                # 出错时返回 500 与错误信息
                ns.abort(500, f"Inference failed: {e}")
                return

            # 附加返回使用的权重文件路径，方便排查
            out_with_meta = dict(out)  # 复制字典以附加元信息
            out_with_meta["ckpt_used"] = ckpt_path  # 记录实际使用的权重路径
            return out_with_meta, 200  # 返回结果与HTTP状态码

    # 把命名空间注册到外部传入的 Api 实例
    api.add_namespace(ns, path="/readthrough")  # 指定路径前缀 /readthrough
