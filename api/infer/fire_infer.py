# -*- coding: utf-8 -*-
# @Time    : 2025/6/5 15:54:21
# @Author  : 陈澔麟
# @File    : mindie_infer.py

import os.path

from api.infer.Triton_model.triton_client import triton_inference


class fire_infer(object):
    def __init__(self, base_url: str, model_name: str, request_id: str = None, message: list = [],
                 mode: str = 'infer'):
        self.message = message
        self.request_id = request_id
        self.model_name = model_name
        self.mode = mode
        self.base_url = base_url + "/chat/completions"
        self.tritonServer = triton_inference(os.path.join(os.getenv("NEXUSAI_HOME"),"api","infer","Triton_model","weights"),
                                urls=[os.getenv("TRITON_SERVER")])

    def infer(self, prompt: str, question: str, file=None):
        result = self.tritonServer.fire_run("fastvlm_fire", file)
        if len(result) == 0:
            return "无"
        result_to_return = result[0]
        result = result_to_return.parames_vector['idcard']
        if result =="无":
            return "无"
        elif "火星" in result:
            return "动火作业-大模型"
        elif "火" in result:
            return "火情检测-大模型"
        elif "吸烟" in result:
            return "吸烟检测-大模型"
        elif "烟" in result:
            return "烟雾检测-大模型"
        return result

