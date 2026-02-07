# -*- coding: utf-8 -*-  # UTF-8
"""
对齐相关路由：/align 与 /align/export
"""  # 模块说明

from flask_restx import Namespace, Resource, fields  # RESTX 组件
from flask import request  # 获取请求体
from pythonscript.tRNAs_flat_exporter import export_template_csv  # 导出模版CSV
from services.alignment_wrapper import perform_full_alignment  # 对齐包装

def register_align_namespace(api) -> None:
    """在传入的Api对象上注册对齐命名空间"""  # 函数说明
    ns = Namespace("align", description="Sequence alignment operations")  # 创建命名空间

    # 定义请求模型（Swagger 展示）
    align_req = ns.model("AlignRequest", {
        "target_seq": fields.String(required=True, description="Target tRNA sequence (A/U/G/C).", example="GGU..."),  # 目标序列
        "anticode": fields.String(description="Anticodon triplet (optional).", example="ACG", default=""),  # 反密码子
        "use_llm": fields.Boolean(description="Whether to prioritize LLM for alignment.", default=True),  # 是否优先LLM
    })  # 结束请求模型定义

    # 定义响应模型（Swagger 展示）
    align_res = ns.model("AlignResponse", {
        "template_name": fields.String(description="Name of the matched tRNA template."),  # 模板名称
        "csv_content": fields.String(description="Two-line CSV text: numbering line + aligned sequence line"),  # 两行CSV
    })  # 结束响应模型定义

    @ns.route("/")  # POST /align/
    class Align(Resource):  # REST 资源类
        @ns.expect(align_req, validate=True)  # 声明请求体与校验
        @ns.response(200, "Success", align_res)  # 成功响应描述
        @ns.response(400, "Bad Request")  # 参数错误
        @ns.response(500, "Internal Server Error")  # 服务错误
        def post(self):  # 处理POST请求
            """Perform six-region alignment and return CSV content."""  # 接口说明
            data = request.get_json(force=True) or {}  # 取JSON体
            target_seq = (data.get("target_seq") or "").strip()  # 序列必填
            if not target_seq:  # 空序列报错
                ns.abort(400, "Field `target_seq` is required and must be non-empty.")  # 抛400
            anticode = (data.get("anticode") or "").strip()  # 可选反密码子
            use_llm = bool(data.get("use_llm", False))  # 是否优先LLM

            try:  # 调用对齐
                tpl_name, csv_raw = perform_full_alignment(
                    target_seq=target_seq,  # 序列
                    output_csv_path="/dev/null",  # 不落盘
                    anticode=anticode,  # 反密码子
                    use_llm=use_llm,  # 优先LLM
                )  # 得到(模板, 两行CSV)
            except Exception as e:  # 捕获异常
                ns.abort(500, f"Alignment failed: {e}")  # 抛500

            return {"template_name": tpl_name, "csv_content": csv_raw}, 200  # 返回JSON

    # 导出子命名空间（/align/export）
    export_ns = Namespace("align/export", description="Export template CSV")  # 新命名空间

    export_res = export_ns.model("ExportResponse", {
        "template_name": fields.String(description="Template name"),  # 模板名
        "csv_content": fields.String(description="Two-line CSV text"),  # 两行CSV
    })  # 响应模型

    @export_ns.route("/")  # GET /align/export/
    class Export(Resource):  # 资源类
        @export_ns.doc(params={"template_name": "Template name (seqname) to export"})  # 文档参数描述
        @export_ns.response(200, "Success", export_res)  # 成功响应
        @export_ns.response(400, "Missing or invalid parameter")  # 参数问题
        @export_ns.response(404, "Template not found")  # 未找到
        def get(self):  # 处理GET
            """Export the two-line CSV text for a given template_name."""  # 接口说明
            template_name = (request.args.get("template_name") or "").strip()  # 读取查询参数
            if not template_name:  # 缺参
                export_ns.abort(400, "Missing query parameter `template_name`")  # 抛400
            try:  # 尝试导出
                csv_raw = export_template_csv(template_name)  # 调用你的函数
            except Exception as e:  # 出错
                export_ns.abort(500, f"Export error: {e}")  # 抛500
            if csv_raw is None:  # 未找到
                export_ns.abort(404, f"Template not found: {template_name}")  # 抛404
            return {"template_name": template_name, "csv_content": csv_raw}, 200  # 返回JSON

    api.add_namespace(ns)  # 把 /align 挂到 Api
    api.add_namespace(export_ns)  # 把 /align/export 挂到 Api
