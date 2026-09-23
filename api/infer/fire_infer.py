# # -*- coding: utf-8 -*-
# # @Time    : 2025/6/5 15:54:21
# # @Author  : 陈澔麟
# # @File    : mindie_infer.py

import os.path
import traceback
import ast
from api.infer.Triton_model.triton_client import triton_inference
import json
from tools.logger_tools import CangQiong_Smart_Vllm_logger as logger_vllm


class fire_infer(object):
    def __init__(self, base_url: str, model_name: str, request_id: str = None, message: list = [],
                 mode: str = 'infer'):
        self.message = message
        self.request_id = request_id
        self.model_name = model_name
        self.mode = mode
        self.base_url = base_url + "/chat/completions"
        self.tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[os.getenv("VLLM_TRITON_SERVER")])
    def _frame_infer(self, prompt: str, question: str, img_list: list):
        """
        多帧推理：跳过 YOLO，直接将已裁剪好的多帧图片列表送入 VLM 模型分析。
        :param prompt: 系统提示词
        :param question: 问题
        :param img_list: 裁剪后的图片列表 (numpy.ndarray)
        :return: 识别结果列表
        """
        return_result = []
        try:
            if not img_list:
                return []

            # 直接调用 VLM 模型 (cangqiong_4b) 处理多帧
            result_fire = self.tritonServer.run(
                "cangqiong_4b",
                img_list,
                label_to_detect={'desc_fire': {'iou': 1, 'conf': 1}}
            )

            if len(result_fire) == 0:
                return []

            analyse_desc = result_fire[0].parames_vector.get('analyse_desc', '')
            logger_vllm.info(f"frame_infer analyse_desc: {analyse_desc}")

            # 解析 JSON 结果（兼容旧 Python 字典格式）
            parsed_desc = {}
            try:
                parsed_desc = json.loads(analyse_desc)
            except Exception:
                # 仅解析字面量，禁止执行模型输出
                try:
                    parsed_desc = ast.literal_eval(analyse_desc)
                except Exception:
                    parsed_desc = {"raw": analyse_desc}

            logger_vllm.info(f"frame_infer parsed: {parsed_desc}")

            # 构建返回结果
            is_alarm = parsed_desc.get("has_fire_smoke", False)
            reason = parsed_desc.get("reason", analyse_desc)

            return_result = [{
                "desc": reason,
                "has_fire_smoke": is_alarm
            }]

        except Exception as e:
            logger_vllm.error(f"frame_infer 异常: {traceback.format_exc()}")

        return return_result

