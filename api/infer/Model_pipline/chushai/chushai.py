# -*- coding: utf-8 -*-
"""
@File    : chushai.py
@Time    : 2026/9/18 10:03
@Author  : 陈冲
@Description : 该文件的功能描述
@Version : 1.0
"""
from api.infer.Utils.class_info import ModelClass
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

class Model(ModelClass):

    def execute(self):
        try:
            boundingboxs = []
            for info in self.logicResult:
                info.classname = self.logicModelName
                boundingboxs.append(info)
            return boundingboxs,[]
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)
            return [], []
