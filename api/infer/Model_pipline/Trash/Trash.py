# -*- coding: utf-8 -*-
"""
@File    : Trash.py
@Time    : 2026/8/25 17:16
@Author  : 陈冲
@Description : 大片垃圾
@Version : 1.0
"""

from api.infer.Utils.class_info import ModelClass
from tools.logger_tools import CangQiong_Smart_Model_logger as logger

class Model(ModelClass):

    def execute(self):
        try:
            boundingboxs = []
            # 不再需要labels，不再调用analyseRun二次识别
            for info in self.logicResult:
                # 计算框宽高
                w = info.x2 - info.x1
                h = info.y2 - info.y1
                # 判断是否大于100*100
                if w > 100 and h > 100:
                    info.classname = self.logicModelName
                    boundingboxs.append(info)
            return boundingboxs, []
        except Exception as e:
            logger.error(f"{self.logicModelName} {e}")
            logger.error(e.__traceback__.tb_frame.f_globals["__file__"])
            logger.error(e.__traceback__.tb_lineno)

        return [], []

